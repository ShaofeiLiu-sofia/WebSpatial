import json
import os
import time
import argparse
from pathlib import Path

def clean_circular_refs(obj, seen=None):
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return None

    if isinstance(obj, dict):
        seen.add(obj_id)
        cleaned = {}
        for k, v in obj.items():
            try:
                cleaned[k] = clean_circular_refs(v, seen.copy()) if isinstance(v, (dict, list)) else v
            except Exception:
                cleaned[k] = None
        return cleaned
    elif isinstance(obj, list):
        seen.add(obj_id)
        cleaned = []
        for item in obj:
            try:
                cleaned.append(clean_circular_refs(item, seen.copy()) if isinstance(item, (dict, list)) else item)
            except Exception:
                cleaned.append(None)
        return cleaned
    else:
        return obj

class CircularReferenceHandler(json.JSONEncoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = set()

    def default(self, obj):
        if isinstance(obj, dict):
            obj_id = id(obj)
            if obj_id in self.seen:
                return None
            self.seen.add(obj_id)
        return super().default(obj)

    def encode(self, o):
        self.seen = set()
        return super().encode(o)

_JS_SERIALIZER = """
(arg) => {
    const { scene_path, sample_limit } = arg;
    const scene = eval(scene_path);
    if (!scene) return { error: `Scene not found at path: ${scene_path}` };

    const safeVec3 = (obj) => {
        if (!obj) return null;
        if ('toArray' in obj && typeof obj.toArray === 'function') return obj.toArray();
        if ('x' in obj && 'y' in obj && 'z' in obj) return [obj.x, obj.y, obj.z];
        return null;
    };

    // Collect world-space positions for all Mesh objects
    const meshPositions = [];
    scene.traverse(obj => {
        if (obj.isMesh) {
            let worldPos = [0, 0, 0];
            try {
                if (obj.matrixWorld && obj.matrixWorld.elements) {
                    const me = obj.matrixWorld.elements;
                    worldPos = [me[12], me[13], me[14]];
                }
            } catch(e) {}
            meshPositions.push({
                uuid: obj.uuid || null,
                name: obj.name || "",
                type: obj.geometry ? obj.geometry.type : obj.type,
                position: safeVec3(obj.position),
                worldPosition: worldPos
            });
        }
    });

    // Prefer scene.toJSON() when available
    let json = null;
    if (scene.toJSON && typeof scene.toJSON === 'function') {
        try {
            json = scene.toJSON();
            json._exportMode = "toJSON";
        } catch(e) {
            console.warn("scene.toJSON() failed, falling back to manual traversal:", e);
        }
    }

    // Inject world positions into the toJSON output
    if (json && json.object && Array.isArray(json.object.children)) {
        const posMap = {};
        meshPositions.forEach(p => { posMap[p.uuid] = p; });

        const injectPositions = (node) => {
            if (node.uuid && posMap[node.uuid]) {
                node.position      = posMap[node.uuid].position;
                node.worldPosition = posMap[node.uuid].worldPosition;
                node.type          = posMap[node.uuid].type;
            }
            if (node.children) node.children.forEach(injectPositions);
        };
        injectPositions(json.object);
        json._meshPositions = meshPositions;
        return json;
    }

    // Fallback: return mesh positions only
    return {
        _exportMode: "manual_traverse",
        meshes: meshPositions
    };
}
"""

def export_scene_web(
    url: str,
    outpath: str = "scene.json",
    scene_path: str = "window.scene",
    sample_limit=None,
    timeout_s: int = 15,
    min_mesh_count: int = 6,
    headless: bool = False,
    save_screenshot: bool = True,
) -> dict:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "playwright is required: pip install playwright && playwright install chromium"
        )

    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=[
                "--use-gl=egl",
                "--enable-webgl",
                "--ignore-gpu-blocklist",
                "--disable-gpu-sandbox",
            ],
        )
        page = browser.new_page()
        page.goto(url)
        time.sleep(5)

        t0 = time.time()
        while True:
            ready = page.evaluate(f"!!({scene_path})")
            if ready:
                break
            if time.time() - t0 > timeout_s:
                raise RuntimeError(
                    f"Scene not ready after {timeout_s}s at path: {scene_path}"
                )
            time.sleep(0.3)

        t0 = time.time()
        while True:
            mesh_count = page.evaluate("""
            () => {
                const meshes = [];
                window.scene.traverse(obj => { if (obj.isMesh) meshes.push(obj); });
                return meshes.length;
            }
            """)
            if mesh_count >= min_mesh_count:
                break
            if time.time() - t0 > 5:
                print(
                    f"Warning: only {mesh_count} mesh(es) found "
                    f"(expected >= {min_mesh_count}), continuing anyway."
                )
                break
            time.sleep(0.3)

        page.wait_for_function(
            """
            () => window.scene &&
                  window.scene.isScene === true &&
                  document.querySelector('canvas')
            """,
            timeout=30000,
        )
        time.sleep(2)

        page.evaluate("""
        (async () => {
            for (let i = 0; i < 5; i++) {
                await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
            }
        })();
        """)

        scene_json = page.evaluate(_JS_SERIALIZER, {
            "scene_path": scene_path,
            "sample_limit": sample_limit,
        })
        scene_json = clean_circular_refs(scene_json)

        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(scene_json, f, indent=2, ensure_ascii=False, cls=CircularReferenceHandler)

        if save_screenshot:
            screenshot_path = str(Path(outpath).with_suffix(".png"))
            page.screenshot(path=screenshot_path)
            print(f"Screenshot saved: {screenshot_path}")

        browser.close()

    print(f"[Web] Scene exported: {outpath}")
    return scene_json

def _extract_gltf_material(mat) -> dict:
    info = {}
    if mat.name:
        info["name"] = mat.name
    if mat.pbrMetallicRoughness:
        pbr = mat.pbrMetallicRoughness
        if pbr.baseColorFactor:
            info["baseColorFactor"] = list(pbr.baseColorFactor)
        if pbr.metallicFactor is not None:
            info["metallicFactor"] = pbr.metallicFactor
        if pbr.roughnessFactor is not None:
            info["roughnessFactor"] = pbr.roughnessFactor
    return info

def export_scene_gltf(
    gltf_path: str,
    outpath: str = "scene.json",
) -> dict:
    try:
        from pygltflib import GLTF2
    except ImportError:
        raise ImportError("pygltflib is required: pip install pygltflib")

    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)

    gltf = GLTF2().load(gltf_path)
    result = {"nodes": [], "_exportMode": "gltf", "_source": str(gltf_path)}

    for i, node in enumerate(gltf.nodes):
        node_info = {
            "index": i,
            "name": node.name or f"node_{i}",
            "mesh": node.mesh,
            "translation": list(node.translation) if node.translation else [0.0, 0.0, 0.0],
            "rotation": list(node.rotation) if node.rotation else [0.0, 0.0, 0.0, 1.0],
            "scale": list(node.scale) if node.scale else [1.0, 1.0, 1.0],
            "children": list(node.children) if node.children else [],
            "materials": [],
        }

        if node.mesh is not None:
            mesh = gltf.meshes[node.mesh]
            node_info["meshName"] = mesh.name or ""
            for prim in mesh.primitives:
                if prim.material is not None:
                    mat = gltf.materials[prim.material]
                    node_info["materials"].append(_extract_gltf_material(mat))

        result["nodes"].append(node_info)

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[GLTF] Scene exported: {outpath}")
    return result

def auto_export(
    url: str = None,
    gltf_path: str = None,
    outpath: str = "scene.json",
    save_screenshot: bool = True,
    **kwargs,
) -> dict:
    provided = sum([url is not None, gltf_path is not None])
    if provided != 1:
        raise ValueError("Provide exactly one of: url, gltf_path")

    if url is not None:
        return export_scene_web(url=url, outpath=outpath, save_screenshot=save_screenshot, **kwargs)
    else:
        return export_scene_gltf(gltf_path=gltf_path, outpath=outpath)

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified scene info extraction tool (Web / GLTF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["web", "gltf", "auto"], default="auto",
        help="Export mode: web=Playwright, gltf=GLB/GLTF file, auto=infer from flags (default)",
    )

    parser.add_argument("--url", default=None,
                        help="[web] Target page URL")
    parser.add_argument("--scene_path", default="window.scene",
                        help="[web] JS path to the scene object (default: window.scene)")
    parser.add_argument("--timeout", type=int, default=15,
                        help="[web] Seconds to wait for the scene to be ready (default: 15)")
    parser.add_argument("--min_mesh", type=int, default=6,
                        help="[web] Minimum expected mesh count (default: 6)")
    parser.add_argument("--headless", action="store_true",
                        help="[web] Run browser in headless mode")

    parser.add_argument("--gltf", default=None, dest="gltf_path",
                        help="[gltf] Path to the .glb or .gltf file")

    parser.add_argument("--out", default="scene.json",
                        help="Output JSON file path (default: scene.json)")
    parser.add_argument("--no_screenshot", action="store_true",
                        help="[web] Do not save a screenshot (default: save screenshot)")
    return parser

if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    mode = args.mode

    if mode == "auto":
        if args.url:
            mode = "web"
        elif args.gltf_path:
            mode = "gltf"
        else:
            parser.error("--mode auto requires either --url or --gltf")

    if mode == "web":
        if not args.url:
            parser.error("--mode web requires --url")
        export_scene_web(
            url=args.url,
            outpath=args.out,
            scene_path=args.scene_path,
            timeout_s=args.timeout,
            min_mesh_count=args.min_mesh,
            headless=args.headless,
            save_screenshot=not args.no_screenshot,
        )
    elif mode == "gltf":
        if not args.gltf_path:
            parser.error("--mode gltf requires --gltf")
        export_scene_gltf(gltf_path=args.gltf_path, outpath=args.out)

