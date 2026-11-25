from __future__ import annotations

import json
from typing import Any

UNIVERSAL_RAW_STATE_JS = r"""
async () => {
    const seen = new WeakSet();
    const scalar = (value) => value === null || ["string", "number", "boolean"].includes(typeof value);
    const safeValue = (value, depth = 0) => {
        if (scalar(value)) return value;
        if (typeof value === "undefined" || typeof value === "function" || typeof value === "symbol") return undefined;
        if (depth > 3) return "[MaxDepth]";
        if (typeof value !== "object") return String(value);
        if (seen.has(value)) return "[Circular]";
        seen.add(value);
        if (Array.isArray(value)) {
            return value.slice(0, 80).map((item) => safeValue(item, depth + 1)).filter((item) => typeof item !== "undefined");
        }
        const out = {};
        for (const key of Object.keys(value).slice(0, 80)) {
            if (key.startsWith("_") && depth > 0) continue;
            try {
                const child = safeValue(value[key], depth + 1);
                if (typeof child !== "undefined") out[key] = child;
            } catch (e) {}
        }
        return out;
    };
    const dom = Array.from(document.querySelectorAll("body *")).slice(0, 600).map((el, index) => {
        const rect = el.getBoundingClientRect();
        const attrs = {};
        for (const attr of Array.from(el.attributes || [])) {
            if (/^(class|id|data-|aria-|role|type|value|name|title)/.test(attr.name)) attrs[attr.name] = attr.value;
        }
        return {
            index,
            tag: el.tagName.toLowerCase(),
            text: (el.innerText || el.textContent || "").trim().slice(0, 120),
            attributes: attrs,
            rect: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
        };
    });
    const canvases = Array.from(document.querySelectorAll("canvas")).map((canvas, index) => {
        const rect = canvas.getBoundingClientRect();
        return {
            index,
            width: canvas.width,
            height: canvas.height,
            rect: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
        };
    });
    const storage = {};
    for (const storeName of ["localStorage", "sessionStorage"]) {
        try {
            const store = window[storeName];
            storage[storeName] = {};
            for (let i = 0; i < Math.min(store.length, 100); i += 1) {
                const key = store.key(i);
                storage[storeName][key] = store.getItem(key);
            }
        } catch (e) {}
    }
    const globalCandidates = {};
    const names = Object.getOwnPropertyNames(window).filter((name) => {
        if (/^(webkit|chrome|CSS|HTML|SVG|Event|Mouse|Keyboard|Pointer|Audio|Image|Intl|Math|JSON|Array|Object|String|Number|Boolean|Date|RegExp|Promise|Reflect|Proxy|Symbol|Map|Set|Weak)/.test(name)) return false;
        if (/^(on|__|performance|document|window|self|top|parent|frames|location|history|navigator|screen|visualViewport|crypto|indexedDB|localStorage|sessionStorage)/.test(name)) return false;
        return true;
    });
    for (const name of names.slice(0, 250)) {
        try {
            const value = window[name];
            if (typeof value === "function" || typeof value === "undefined") continue;
            const serialized = safeValue(value);
            if (typeof serialized !== "undefined") globalCandidates[name] = serialized;
        } catch (e) {}
    }
    const scriptTexts = [];
    for (const script of Array.from(document.scripts || []).slice(0, 80)) {
        try {
            if (script.textContent && script.textContent.trim()) {
                scriptTexts.push({src: script.src || "", text: script.textContent.slice(0, 200000)});
            } else if (script.src && script.src.startsWith(location.origin)) {
                const response = await fetch(script.src);
                scriptTexts.push({src: script.src, text: (await response.text()).slice(0, 200000)});
            }
        } catch (e) {}
    }
    scriptTexts.sort((a, b) => {
        const score = (entry) => (entry.src.includes("/vendor/") ? 10 : 0) + (entry.src.includes("/js/") ? -5 : 0);
        return score(a) - score(b);
    });
    const identifierPatterns = [
        /\b(?:let|const|var|function|class)\s+([A-Za-z_$][A-Za-z0-9_$]*)/g,
        /\bwindow\.([A-Za-z_$][A-Za-z0-9_$]*)/g,
        /(?:^|[;\n\r])\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*=/g
    ];
    const identifiers = new Set();
    for (const entry of scriptTexts) {
        for (const pattern of identifierPatterns) {
            let match = null;
            while ((match = pattern.exec(entry.text)) && identifiers.size < 800) {
                identifiers.add(match[1]);
            }
        }
    }
    for (const name of identifiers) {
        if (name in globalCandidates) continue;
        try {
            const value = eval(name);
            if (typeof value === "function" || typeof value === "undefined") continue;
            const serialized = safeValue(value);
            if (typeof serialized !== "undefined") globalCandidates[name] = serialized;
        } catch (e) {}
    }
    return {
        schema: "universal_raw_browser_state_v1",
        url: location.href,
        title: document.title,
        dom,
        canvases,
        storage,
        scriptIdentifiers: Array.from(identifiers).slice(0, 400),
        globals: globalCandidates,
    };
}
"""

async def observe_raw_universal(page) -> dict[str, Any]:
    return await page.evaluate(UNIVERSAL_RAW_STATE_JS)

def compact_raw_samples(raw_states: list[dict[str, Any]], max_chars: int = 60000) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    used = 0
    for state in raw_states:
        candidate = {
            "schema": state.get("schema"),
            "title": state.get("title"),
            "dom": state.get("dom", [])[:120],
            "canvases": state.get("canvases", []),
            "storage": state.get("storage", {}),
            "globals": state.get("globals", {}),
        }
        text = json.dumps(candidate, ensure_ascii=False)
        if used + len(text) > max_chars and compact:
            break
        compact.append(candidate)
        used += len(text)
    return compact

