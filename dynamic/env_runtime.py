from __future__ import annotations

import asyncio
import contextlib
import math
import socket
import threading
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

from universal_raw_state import observe_raw_universal

@dataclass
class DynamicObservation:
    index: int
    action_count: int
    image: str
    state: Any

@dataclass
class DynamicStep:
    index: int
    action: str
    state_before: Any
    state_after: Any

@dataclass
class DynamicTrace:
    env: str
    scenario_id: str
    split: str
    seed: int
    observations: list[DynamicObservation] = field(default_factory=list)
    steps: list[DynamicStep] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    visual_rmsdiff: float | None = None
    scenario_file: str | None = None
    capture_mode: str = "replay"
    finish_reason: str = "completed_actions"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame_record(self) -> dict[str, Any]:
        record = {
            "env": self.env,
            "scenario_id": self.scenario_id,
            "split": self.split,
            "seed": self.seed,
            "images": [obs.image for obs in self.observations],
            "states": [obs.state for obs in self.observations],
            "action_counts_at_frames": [obs.action_count for obs in self.observations],
            "actions": self.actions,
            "steps": [
                {
                    "index": step.index,
                    "action": step.action,
                    "state_before": step.state_before,
                    "state_after": step.state_after,
                }
                for step in self.steps
            ],
            "visual_rmsdiff": self.visual_rmsdiff,
            "capture_mode": self.capture_mode,
            "finish_reason": self.finish_reason,
            "trace_schema": "dynamic_env_trace_v1",
        }
        if self.scenario_file:
            record["scenario_file"] = self.scenario_file
        record.update(self.metadata)
        return record

class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

@contextlib.contextmanager
def serve_directory(directory: Path):
    port = free_port()
    handler = partial(QuietHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

def seed_script(seed: int) -> str:
    return f"""
        (() => {{
            let state = {seed} >>> 0;
            function next() {{
                state = (1664525 * state + 1013904223) >>> 0;
                return state / 4294967296;
            }}
            Math.random = next;
            try {{
                localStorage.clear();
                sessionStorage.clear();
            }} catch (e) {{}}
            window.__spatialGymSeed = {seed};
            window.__spatialGymReplay = true;
        }})();
    """

def image_rmsdiff(before: Path, after: Path) -> float:
    with Image.open(before) as img_a, Image.open(after) as img_b:
        diff = ImageChops.difference(img_a.convert("RGB"), img_b.convert("RGB"))
        stat = ImageStat.Stat(diff)
        sq = sum(value ** 2 for value in stat.rms) / len(stat.rms)
        return math.sqrt(sq)

async def wait_until_app_ready(page, spec: dict[str, Any], timeout_ms: int = 10000) -> None:
    ready_js = spec.get("ready_js")
    if not ready_js:
        return
    deadline = asyncio.get_running_loop().time() + timeout_ms / 1000
    while True:
        try:
            if await page.evaluate(ready_js):
                return
        except Exception:
            pass
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError("dynamic app did not become ready before capture")
        await page.wait_for_timeout(100)

async def fit_viewport_to_app(page, spec: dict[str, Any]) -> None:
    base = spec.get("viewport", {})
    size = await page.evaluate(
        """
        () => {
            const rects = Array.from(document.querySelectorAll("canvas, #canvas, #game, .game, body"))
                .map((el) => {
                    const rect = el.getBoundingClientRect();
                    return {
                        width: Math.ceil(rect.right + window.scrollX),
                        height: Math.ceil(rect.bottom + window.scrollY)
                    };
                });
            return rects.reduce(
                (acc, rect) => ({
                    width: Math.max(acc.width, rect.width),
                    height: Math.max(acc.height, rect.height)
                }),
                {
                    width: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
                    height: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)
                }
            );
        }
        """
    )
    width = max(int(base.get("width", 800)), int(size.get("width") or 0))
    height = max(int(base.get("height", 800)), int(size.get("height") or 0))
    await page.set_viewport_size({"width": width, "height": height})
    resize_js = spec.get("resize_js")
    if resize_js:
        await page.evaluate(resize_js)

class DynamicEnvRuntime:

    def __init__(self, page, url: str, spec: dict[str, Any], frame_dir: Path):
        self.page = page
        self.url = url
        self.spec = spec
        self.frame_dir = frame_dir
        self.trace: DynamicTrace | None = None

    async def install_local_route(self) -> None:
        async def route_local_only(route):
            request_url = route.request.url
            if request_url.startswith(self.url) or request_url.startswith("data:"):
                await route.continue_()
            else:
                await route.abort()

        await self.page.route("**/*", route_local_only)

    async def reset(self, scenario_id: str, split: str, seed: int, capture_mode: str = "replay") -> None:
        await self.page.add_init_script(seed_script(seed))
        await self.page.goto(self.url, wait_until="load")
        await self.page.evaluate(self.spec["start_js"])
        await self.page.wait_for_timeout(int(self.spec["warmup_ms"]))
        await wait_until_app_ready(self.page, self.spec)
        prepared = await self.page.evaluate(self.spec["prepare_visual_state_js"])
        if not prepared:
            raise RuntimeError(f"failed to prepare visible app state for {scenario_id}")
        await self.page.wait_for_timeout(50)
        await fit_viewport_to_app(self.page, self.spec)
        self.trace = DynamicTrace(
            env=str(self.spec.get("env_name", "")),
            scenario_id=scenario_id,
            split=split,
            seed=seed,
            capture_mode=capture_mode,
        )

    async def observe(self, action_count: int, label: str | None = None) -> DynamicObservation:
        if self.trace is None:
            raise RuntimeError("reset must be called before observe")
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        suffix = label or f"T{len(self.trace.observations):03d}"
        frame_path = self.frame_dir / f"{self.trace.scenario_id}_{suffix}.png"
        await self.page.evaluate(self.spec.get("render_js", "() => true"))
        await self.page.screenshot(path=str(frame_path))
        observation = DynamicObservation(
            index=len(self.trace.observations),
            action_count=action_count,
            image=str(frame_path.resolve()),
            state=await self._observe_state(),
        )
        self.trace.observations.append(observation)
        return observation

    async def step(self, action: str) -> DynamicStep:
        if self.trace is None:
            raise RuntimeError("reset must be called before step")
        state_before = await self._observe_state()
        if "action_steps" in self.spec:
            step = int(self.spec["action_steps"][action])
            await self.page.evaluate("(step) => { MainHex.rotate(step); render(); }", step)
        elif "action_vectors" in self.spec:
            axis = self.spec["action_vectors"][action]
            await self.page.evaluate("(axis) => { if (typeof onMoveKey === 'function') onMoveKey(axis); }", axis)
        else:
            key = self.spec["key_map"][action]
            await self.page.keyboard.press(key)
        await self.page.wait_for_timeout(int(self.spec["action_wait_ms"]))
        state_after = await self._observe_state()
        dynamic_step = DynamicStep(
            index=len(self.trace.steps),
            action=action,
            state_before=state_before,
            state_after=state_after,
        )
        self.trace.steps.append(dynamic_step)
        self.trace.actions.append(action)
        return dynamic_step

    async def _observe_state(self) -> Any:
        if self.spec.get("state_mode") == "universal":
            return await observe_raw_universal(self.page)
        return await self.page.evaluate(self.spec["state_js"])

    async def finalize(self, min_rmsdiff: float = 1.0) -> DynamicTrace:
        if self.trace is None:
            raise RuntimeError("reset must be called before finalize")
        await self.page.wait_for_timeout(int(self.spec["settle_ms"]))
        await self.page.evaluate(self.spec.get("render_js", "() => true"))
        if len(self.trace.observations) >= 2:
            before = Path(self.trace.observations[0].image)
            after = Path(self.trace.observations[-1].image)
            self.trace.visual_rmsdiff = image_rmsdiff(before, after)
            if self.trace.visual_rmsdiff < min_rmsdiff:
                raise RuntimeError(
                    f"{self.trace.scenario_id} replay produced no visible transform; "
                    f"rmsdiff={self.trace.visual_rmsdiff:.4f}"
                )
        return self.trace

