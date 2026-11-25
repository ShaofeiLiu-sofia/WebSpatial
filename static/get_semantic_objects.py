import json
import os
import argparse
import numpy as np

def _add_name_indices(objects: list) -> list:

    type_counts = {}
    for obj in objects:
        obj_type = obj.get("type") or "unknown"
        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

    type_indices = {}
    result = []

    for obj in objects:
        obj_type = obj.get("type") or "unknown"

        new_obj = obj.copy()

        if type_counts[obj_type] > 1:
            if obj_type not in type_indices:
                type_indices[obj_type] = 0
            type_indices[obj_type] += 1
            new_obj["name"] = f"{obj_type}_{type_indices[obj_type]}"
        else:

            new_obj["name"] = obj.get("name", obj_type)

        result.append(new_obj)

    return result

def _int_to_hex(value: int) -> str:
    return "#{:06x}".format(int(value) & 0xFFFFFF)

def _factor_to_hex(factor) -> str:
    r, g, b = factor[:3]
    return "#{:02x}{:02x}{:02x}".format(
        min(255, int(r * 255)),
        min(255, int(g * 255)),
        min(255, int(b * 255)),
    )

def _name_to_hex(name: str):
    if not name:
        return None
    try:
        hex_part = name.split(".")[0]
        if len(hex_part) == 6:
            int(hex_part, 16)
            return f"#{hex_part.lower()}"
    except Exception:
        pass
    return None

def _trs_to_matrix(t, r, s) -> np.ndarray:
    tx, ty, tz = t
    qx, qy, qz, qw = r
    sx, sy, sz = s

    x2, y2, z2 = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    rot = np.array([
        [1 - 2*(y2+z2),   2*(xy-wz),     2*(xz+wy),   0],
        [2*(xy+wz),       1 - 2*(x2+z2), 2*(yz-wx),   0],
        [2*(xz-wy),       2*(yz+wx),     1 - 2*(x2+y2), 0],
        [0,               0,             0,            1],
    ], dtype=float)

    mat = np.eye(4, dtype=float)
    mat[:3, :3] = rot[:3, :3] * np.array([[sx], [sy], [sz]])
    mat[:3, 3] = [tx, ty, tz]
    return mat

def _compute_gltf_world_matrices(nodes: list) -> dict:

    parent = {}
    for node in nodes:
        idx = node["index"]
        for child_idx in node.get("children", []):
            parent[child_idx] = idx

    cache = {}

    def world_of(idx):
        if idx in cache:
            return cache[idx]
        node = nodes[idx]
        t = node.get("translation") or [0.0, 0.0, 0.0]
        r = node.get("rotation") or [0.0, 0.0, 0.0, 1.0]
        s = node.get("scale") or [1.0, 1.0, 1.0]
        local = _trs_to_matrix(t, r, s)
        if idx in parent:
            result = world_of(parent[idx]) @ local
        else:
            result = local
        cache[idx] = result
        return result

    for node in nodes:
        world_of(node["index"])

    return cache

def _extract_tojson(data: dict) -> list:
    geometries = {g["uuid"]: g for g in data.get("geometries", [])}
    materials  = {m["uuid"]: m for m in data.get("materials", [])}

    objects = []

    def traverse(node):
        if not node or not isinstance(node, dict):
            return

        if not node.get("visible", True):
            return

        geom_id = node.get("geometry")
        mat_id  = node.get("material")

        if geom_id is not None:
            geom = geometries.get(geom_id, {})
            mat  = materials.get(mat_id, {}) if mat_id else {}

            name = node.get("name") or geom.get("type") or node.get("type") or "unknown"

            shape = geom.get("type") or node.get("type") or "unknown"

            if shape == "BufferGeometry":
                return

            color = None
            raw_color = mat.get("color")
            if raw_color is not None:
                try:
                    color = _int_to_hex(int(raw_color))
                except (ValueError, TypeError):
                    pass

            objects.append({
                "name":          name,
                "type":          shape,
                "color":         color,
                "position":      node.get("position"),
                "worldPosition": node.get("worldPosition"),
            })

        for child in node.get("children", []):
            traverse(child)

    traverse(data.get("object", {}))
    return _add_name_indices(objects)

def _extract_manual(data: dict) -> list:
    objects = []
    for mesh in data.get("meshes", []):
        type_ = mesh.get("type") or "unknown"
        if type_ == "BufferGeometry":
            continue
        objects.append({
            "name":          mesh.get("name") or type_ or "unknown",
            "type":          type_,
            "color":         None,
            "position":      mesh.get("position"),
            "worldPosition": mesh.get("worldPosition"),
        })
    return _add_name_indices(objects)

def _extract_gltf(data: dict) -> list:
    nodes = data.get("nodes", [])
    if not nodes:
        return []

    world_matrices = _compute_gltf_world_matrices(nodes)

    objects = []
    for node in nodes:

        if node.get("mesh") is None:
            continue

        idx   = node["index"]
        name  = node.get("name") or f"node_{idx}"
        shape = node.get("meshName") or name

        if shape == "BufferGeometry":
            continue

        wm = world_matrices.get(idx, np.eye(4))
        world_pos = wm[:3, 3].tolist()

        local_pos = node.get("translation") or [0.0, 0.0, 0.0]

        color = None
        for mat in node.get("materials", []):
            color = _name_to_hex(mat.get("name", ""))
            if color:
                break
            factor = mat.get("baseColorFactor")
            if factor and factor[:3] != [1.0, 1.0, 1.0]:
                color = _factor_to_hex(factor)
                break

        objects.append({
            "name":          name,
            "type":          shape,
            "color":         color,
            "position":      local_pos,
            "worldPosition": world_pos,
        })

    return _add_name_indices(objects)

def get_semantic_objects(scene_json_path: str) -> list:
    with open(scene_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mode = data.get("_exportMode", "")

    if mode == "gltf":
        return _extract_gltf(data)

    if mode == "manual_traverse":
        return _extract_manual(data)

    if mode == "toJSON" or "object" in data:
        return _extract_tojson(data)

    for fn in (_extract_tojson, _extract_manual, _extract_gltf):
        result = fn(data)
        if result:
            return result

    return []

def export_semantic_objects(
    scene_json_path: str,
    outpath: str,
) -> list:
    objects = get_semantic_objects(scene_json_path)
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(objects, f, indent=2, ensure_ascii=False)
    print(f"Extracted {len(objects)} semantic objects → {outpath}")
    return objects

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract semantic_objects.json from a get_scene_info.py scene export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scene", required=True,
        help="Path to the scene JSON file produced by get_scene_info.py",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output path for semantic_objects.json",
    )
    return parser

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    export_semantic_objects(scene_json_path=args.scene, outpath=args.out)

