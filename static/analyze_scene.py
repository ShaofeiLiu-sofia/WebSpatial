import json
import os
import argparse
from collections import defaultdict
from itertools import permutations

_COLOR_NAMES: dict[int, str] = {
    0xFF0000: "red",
    0x00FF00: "green",
    0x0000FF: "blue",
    0xFFFF00: "yellow",
    0x00FFFF: "cyan",
    0xFF00FF: "magenta",
    0xFFFFFF: "white",
    0x000000: "black",
    0xFFA500: "orange",
    0x808080: "gray",
    0x800080: "purple",
    0x964B00: "brown",
    0xADD8E6: "lightblue",
    0x90EE90: "lightgreen",
    0xFFC0CB: "pink",
    0xA52A2A: "brown",
    0xF5F5DC: "beige",
    0x808000: "olive",
    0x008080: "teal",
    0x000080: "navy",
    0xC0C0C0: "silver",
    0xFFD700: "gold",
}

def _parse_color(raw) -> tuple[str, str] | None:
    if raw is None:
        return None

    int_val: int | None = None

    if isinstance(raw, int):
        int_val = raw
    elif isinstance(raw, float):
        int_val = int(raw)
    elif isinstance(raw, str):
        s = raw.strip().lower()
        if s.startswith("#"):
            s = s[1:]
        elif s.startswith("0x"):
            s = s[2:]
        try:
            int_val = int(s, 16)
        except ValueError:
            try:
                int_val = int(s)
            except ValueError:
                return None

    if int_val is None:
        return None

    int_val &= 0xFFFFFF
    r = (int_val >> 16) & 0xFF
    g = (int_val >> 8)  & 0xFF
    b =  int_val        & 0xFF
    hex_literal = f"#{r:02x}{g:02x}{b:02x}"
    color_name  = _COLOR_NAMES.get(int_val, hex_literal)
    return hex_literal, color_name

def compute_color(objects: list) -> list:

    by_type: dict[str, dict[str, dict]] = defaultdict(dict)

    for obj in objects:
        obj_type = obj.get("type") or "unknown"
        parsed = _parse_color(obj.get("color"))
        if not parsed:
            continue
        hex_lit, name = parsed
        bucket = by_type[obj_type]
        if hex_lit not in bucket:
            bucket[hex_lit] = {"hex_literal": hex_lit, "color": name, "count": 0}
        bucket[hex_lit]["count"] += 1

    return [
        {"type": t, "colors": list(cols.values())}
        for t, cols in sorted(by_type.items())
    ]

def compute_shape(objects: list) -> list:
    counter: dict[str, int] = defaultdict(int)
    for obj in objects:
        t = obj.get("type") or "unknown"
        counter[t] += 1

    return [
        {"type": t, "count": c}
        for t, c in sorted(counter.items(), key=lambda x: -x[1])
    ]

def compute_count(objects: list) -> list:

    info: dict[str, dict] = {}
    for obj in objects:
        name  = obj.get("name") or "unknown"
        shape = obj.get("type") or "unknown"
        if name not in info:
            info[name] = {"count": 0, "shape": shape}
        info[name]["count"] += 1

    return [
        {"name": n, "count": d["count"], "shape": d["shape"]}
        for n, d in sorted(info.items(), key=lambda x: -x[1]["count"])
    ]

RELATION_THRESHOLD = 0.1

def _spatial_relations(posA: list, posB: list, threshold: float = RELATION_THRESHOLD) -> list:
    dx = posA[0] - posB[0]
    dy = posA[1] - posB[1]
    dz = posA[2] - posB[2]

    rels = []
    if   dx >  threshold: rels.append("right")
    elif dx < -threshold: rels.append("left")

    if   dy >  threshold: rels.append("above")
    elif dy < -threshold: rels.append("below")

    if   dz >  threshold: rels.append("front")
    elif dz < -threshold: rels.append("behind")

    return rels

def compute_relation(objects: list, threshold: float = RELATION_THRESHOLD) -> list:
    relations = []
    for objA, objB in permutations(objects, 2):
        nameA = objA.get("name")
        nameB = objB.get("name")
        posA  = objA.get("worldPosition")
        posB  = objB.get("worldPosition")

        if not posA or not posB:
            continue

        for rel in _spatial_relations(posA, posB, threshold):
            relations.append({"ObjectA": nameA, "ObjectB": nameB, "relation": rel})

    return relations

def analyze_all(
    objects: list,
    outdir: str,
    threshold: float = RELATION_THRESHOLD,
) -> dict:
    os.makedirs(outdir, exist_ok=True)

    color    = compute_color(objects)
    shape    = compute_shape(objects)
    count    = compute_count(objects)
    relation = compute_relation(objects, threshold)

    results = {
        "color":    color,
        "shape":    shape,
        "count":    count,
        "relation": relation,
    }

    for key, data in results.items():
        path = os.path.join(outdir, f"{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[{key}] {len(data)} entries → {path}")

    return results

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute color / shape / count / relation from semantic_objects.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--objects", required=True,
        help="Path to semantic_objects.json produced by get_semantic_objects.py",
    )
    parser.add_argument(
        "--outdir", required=True,
        help="Directory to write color.json, shape.json, count.json, relation.json",
    )
    parser.add_argument(
        "--threshold", type=float, default=RELATION_THRESHOLD,
        help=f"Distance threshold for spatial relations (default: {RELATION_THRESHOLD})",
    )
    parser.add_argument(
        "--only", nargs="+",
        choices=["color", "shape", "count", "relation"],
        help="Run only the specified analyses (default: all four)",
    )
    return parser

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    with open(args.objects, "r", encoding="utf-8") as f:
        objects = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)

    targets = args.only or ["color", "shape", "count", "relation"]

    dispatch = {
        "color":    lambda: compute_color(objects),
        "shape":    lambda: compute_shape(objects),
        "count":    lambda: compute_count(objects),
        "relation": lambda: compute_relation(objects, args.threshold),
    }

    for key in targets:
        data = dispatch[key]()
        path = os.path.join(args.outdir, f"{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[{key}] {len(data)} entries → {path}")

