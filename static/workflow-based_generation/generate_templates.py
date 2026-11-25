import json
import os
import argparse
from collections import defaultdict

def _load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _hex_to_0x(hex_str: str) -> str:
    if hex_str and hex_str.startswith("#"):
        return "0x" + hex_str[1:]
    return hex_str

def _color_answer(hex_literal: str, color_name: str) -> dict:
    return {"hex_literal": _hex_to_0x(hex_literal), "color": color_name}

def _qa_color(objects: list) -> list:
    qa = []

    name_color: dict[str, tuple] = {}
    color_to_names: dict[str, list] = defaultdict(list)
    color_to_types: dict[str, list] = defaultdict(list)

    COLOR_NAMES = {
        0xff0000: "red", 0x00ff00: "green", 0x0000ff: "blue",
        0xffff00: "yellow", 0x00ffff: "cyan", 0xff00ff: "magenta",
        0xffffff: "white", 0x000000: "black", 0xffa500: "orange",
        0x808080: "gray", 0x800080: "purple", 0x964b00: "brown",
        0xadd8e6: "lightblue", 0x90ee90: "lightgreen", 0xffc0cb: "pink",
    }

    def _resolve_name(raw) -> tuple[str, str] | None:
        if raw is None:
            return None
        if isinstance(raw, int):
            iv = raw & 0xFFFFFF
        elif isinstance(raw, str):
            s = raw.strip().lstrip("#")
            if s.startswith("0x"):
                s = s[2:]
            try:
                iv = int(s, 16)
            except ValueError:
                return None
        else:
            return None
        r, g, b = (iv >> 16) & 0xFF, (iv >> 8) & 0xFF, iv & 0xFF
        hex_lit = f"#{r:02x}{g:02x}{b:02x}"
        name = COLOR_NAMES.get(iv, hex_lit)
        return hex_lit, name

    for obj in objects:
        n = obj.get("name")
        t = obj.get("type") or n
        parsed = _resolve_name(obj.get("color"))
        if n and parsed:
            hex_lit, cname = parsed
            name_color[n] = (hex_lit, cname)
            color_to_names[cname].append(n)
            color_to_types[cname].append(t)

    seen_c1: set = set()
    for obj in objects:
        n = obj.get("name")
        if n and n in name_color and n not in seen_c1:
            hex_lit, cname = name_color[n]
            qa.append({
                "type": "color",
                "question": f"What color is the {n}?",
                "answer": [_color_answer(hex_lit, cname)],
                "workflow_steps": [
                    f"get_object_color: retrieve color value of {n}",
                    "hex_to_color_name: convert hex color to color name",
                    "format answer as list of {hex_literal, color} objects"
                ],
                "execution_template": {
                    "description": "Query object color by name",
                    "steps": [
                        {
                            "step_id": 0,
                            "function": "get_object_color",
                            "parameters": {"object_name": n},
                            "description": f"Retrieve color value of {n}"
                        },
                        {
                            "step_id": 1,
                            "function": "hex_to_color_name",
                            "parameters": {"hex_value": "<result_0>"},
                            "description": "Convert hex to color name"
                        }
                    ],
                    "final_format": "list_of_color_dicts"
                }
            })
            seen_c1.add(n)

    for cname, names in sorted(color_to_names.items()):
        unique = list(dict.fromkeys(names))
        qa.append({
            "type": "color",
            "question": f"Which objects are {cname}?",
            "answer": unique,
            "workflow_steps": [
                f"get_objects_by_color: find all objects with color {cname}",
                "return list of object names"
            ],
            "execution_template": {
                "description": f"Query objects by color ({cname})",
                "steps": [
                    {
                        "step_id": 0,
                        "function": "get_objects_by_color",
                        "parameters": {"color_name": cname},
                        "description": f"Find all objects with color {cname}"
                    }
                ],
                "final_format": "list_of_strings"
            }
        })

    for cname, types in sorted(color_to_types.items()):
        unique_types = list(dict.fromkeys(types))
        qa.append({
            "type": "color",
            "question": f"What is the shape of the {cname} object?",
            "answer": unique_types,
            "workflow_steps": [
                f"get_objects_by_color: find objects with color {cname}",
                "get_object_shape: extract shape of each object",
                "return unique shapes"
            ],
            "execution_template": {
                "description": f"Query shapes of {cname} objects",
                "steps": [
                    {
                        "step_id": 0,
                        "function": "get_objects_by_color",
                        "parameters": {"color_name": cname},
                        "description": f"Find all objects with color {cname}"
                    },
                    {
                        "step_id": 1,
                        "function": "get_objects_by_shape",
                        "note": "For each object from step 0, query its shape using get_object_shape"
                    }
                ],
                "final_format": "list_of_shape_strings"
            }
        })

    return qa

def _qa_shape(objects: list, shape_data: list) -> list:
    qa = []

    name_type: dict[str, str] = {}
    type_to_names: dict[str, list] = defaultdict(list)

    for obj in objects:
        n = obj.get("name")
        t = obj.get("type")
        if n and t:
            name_type[n] = t
            type_to_names[t].append(n)

    seen: set = set()
    for obj in objects:
        n = obj.get("name")
        t = obj.get("type")
        if n and t and n not in seen:
            qa.append({
                "type": "shape",
                "question": f"What shape is {n}?",
                "answer": t,
                "workflow_steps": [
                    f"get_object_shape: retrieve shape type of {n}"
                ],
                "execution_template": {
                    "description": f"Query shape of object {n}",
                    "steps": [
                        {
                            "step_id": 0,
                            "function": "get_object_shape",
                            "parameters": {"object_name": n},
                            "description": f"Retrieve shape type of {n}"
                        }
                    ],
                    "final_format": "string"
                }
            })
            seen.add(n)

    for entry in shape_data:
        t = entry["type"]
        c = entry["count"]
        qa.append({
            "type": "shape",
            "question": f"How many {t}s are there?",
            "answer": str(c),
            "workflow_steps": [
                f"count_by_shape: count all objects with shape {t}"
            ],
            "execution_template": {
                "description": f"Count objects of shape {t}",
                "steps": [
                    {
                        "step_id": 0,
                        "function": "count_by_shape",
                        "parameters": {"shape_type": t},
                        "description": f"Count all objects with shape {t}"
                    }
                ],
                "final_format": "integer_as_string"
            }
        })
        names = list(dict.fromkeys(type_to_names.get(t, [])))
        if names:
            qa.append({
                "type": "shape",
                "question": f"Which objects are {t}?",
                "answer": names,
                "workflow_steps": [
                    f"get_objects_by_shape: find all objects with shape {t}"
                ],
                "execution_template": {
                    "description": f"Query all objects with shape {t}",
                    "steps": [
                        {
                            "step_id": 0,
                            "function": "get_objects_by_shape",
                            "parameters": {"shape_type": t},
                            "description": f"Find all objects with shape {t}"
                        }
                    ],
                    "final_format": "list_of_strings"
                }
            })

    return qa

def _qa_count(objects: list, count_data: list, shape_data: list) -> list:
    qa = []

    total = len(objects)
    unique_types = len({obj.get("type") for obj in objects if obj.get("type")})

    qa.append({
        "type": "count",
        "question": "How many objects are there in the scene?",
        "answer": str(total),
        "workflow_steps": [
            "count_total_objects: count all objects in semantic_objects"
        ],
        "execution_template": {
            "description": "Count total objects in scene",
            "steps": [
                {
                    "step_id": 0,
                    "function": "count_total_objects",
                    "parameters": {},
                    "description": "Count all objects in the scene"
                }
            ],
            "final_format": "integer_as_string"
        }
    })

    qa.append({
        "type": "count",
        "question": "How many different types of objects are there in the scene?",
        "answer": str(unique_types),
        "workflow_steps": [
            "count_unique_types: count distinct shape types"
        ],
        "execution_template": {
            "description": "Count unique object types",
            "steps": [
                {
                    "step_id": 0,
                    "function": "count_unique_types",
                    "parameters": {},
                    "description": "Count unique shape types in scene"
                }
            ],
            "final_format": "integer_as_string"
        }
    })

    for entry in shape_data:
        n = entry["type"]
        c = entry["count"]
        qa.append({
            "type": "count",
            "question": f"How many {n}s are there?",
            "answer": str(c),
            "workflow_steps": [
                f"count_by_shape: count objects with shape {n}"
            ],
            "execution_template": {
                "description": f"Count {n} objects",
                "steps": [
                    {
                        "step_id": 0,
                        "function": "count_by_shape",
                        "parameters": {"shape_type": n},
                        "description": f"Count objects with shape {n}"
                    }
                ],
                "final_format": "integer_as_string"
            }
        })

    COLOR_NAMES = {
        0xff0000: "red", 0x00ff00: "green", 0x0000ff: "blue",
        0xffff00: "yellow", 0x00ffff: "cyan", 0xff00ff: "magenta",
        0xffffff: "white", 0x000000: "black", 0xffa500: "orange",
        0x808080: "gray", 0x800080: "purple", 0x964b00: "brown",
        0xadd8e6: "lightblue", 0x90ee90: "lightgreen", 0xffc0cb: "pink",
    }

    def _parse_color_int(raw):
        if raw is None:
            return None
        if isinstance(raw, int):
            return raw & 0xFFFFFF
        if isinstance(raw, str):
            s = raw.strip().lstrip("#")
            if s.startswith("0x"):
                s = s[2:]
            try:
                return int(s, 16) & 0xFFFFFF
            except ValueError:
                return None
        return None

    def _color_name(raw):
        iv = _parse_color_int(raw)
        if iv is None:
            return None
        r, g, b = (iv >> 16) & 0xFF, (iv >> 8) & 0xFF, iv & 0xFF
        return COLOR_NAMES.get(iv, f"#{r:02x}{g:02x}{b:02x}")

    color_count: dict[str, int] = defaultdict(int)
    type_color_count: dict[tuple, int] = defaultdict(int)
    color_total: dict[str, int] = defaultdict(int)

    for obj in objects:
        cname = _color_name(obj.get("color"))
        t     = obj.get("type") or "unknown"
        if cname:
            color_count[cname] += 1
            type_color_count[(cname, t)] += 1
        color_total["_all"] += 0
        color_total[cname or "_none"] += 1

    for cname, c in sorted(color_count.items()):
        qa.append({
            "type": "count",
            "question": f"How many {cname} objects are there?",
            "answer": str(c),
            "workflow_steps": [
                f"count_objects_by_color: count objects with color {cname}"
            ],
            "execution_template": {
                "description": f"Count {cname} objects",
                "steps": [
                    {
                        "step_id": 0,
                        "function": "count_objects_by_color",
                        "parameters": {"color_name": cname},
                        "description": f"Count objects with color {cname}"
                    }
                ],
                "final_format": "integer_as_string"
            }
        })

    for (cname, t), c in sorted(type_color_count.items()):
        qa.append({
            "type": "count",
            "question": f"How many {cname} {t}s are there?",
            "answer": str(c),
            "workflow_steps": [
                f"count_objects_by_color_and_shape: count {cname} {t}s"
            ],
            "execution_template": {
                "description": f"Count {cname} {t} objects",
                "steps": [
                    {
                        "step_id": 0,
                        "function": "count_objects_by_color_and_shape",
                        "parameters": {"color_name": cname, "shape_type": t},
                        "description": f"Count objects with color {cname} and shape {t}"
                    }
                ],
                "final_format": "integer_as_string"
            }
        })
        qa.append({
            "type": "count",
            "question": f"How many objects are both {cname} and {t}?",
            "answer": str(c),
            "workflow_steps": [
                f"count_objects_by_color_and_shape: count objects matching both criteria"
            ],
            "execution_template": {
                "description": f"Count objects that are {cname} and {t}",
                "steps": [
                    {
                        "step_id": 0,
                        "function": "count_objects_by_color_and_shape",
                        "parameters": {"color_name": cname, "shape_type": t},
                        "description": f"Count objects with color {cname} and shape {t}"
                    }
                ],
                "final_format": "integer_as_string"
            }
        })

    for cname, c in sorted(color_count.items()):
        not_c = total - c
        qa.append({
            "type": "count",
            "question": f"How many objects are not {cname}?",
            "answer": str(not_c),
            "workflow_steps": [
                f"count_objects_not_color: count objects not matching color {cname}"
            ],
            "execution_template": {
                "description": f"Count objects not {cname}",
                "steps": [
                    {
                        "step_id": 0,
                        "function": "count_objects_not_color",
                        "parameters": {"color_name": cname},
                        "description": f"Count objects that are not {cname}"
                    }
                ],
                "final_format": "integer_as_string"
            }
        })

    return qa

def _qa_relation(objects: list, relation_data: list) -> list:
    qa = []

    pair_rels: dict[tuple, list] = defaultdict(list)
    for r in relation_data:
        pair_rels[(r["ObjectA"], r["ObjectB"])].append(r["relation"])

    for (a, b), rels in pair_rels.items():
        unique_rels = list(dict.fromkeys(rels))
        answer = ", ".join(unique_rels)
        qa.append({
            "type": "relation",
            "question": f"Where is {a} relative to {b}?",
            "answer": answer,
            "workflow_steps": [
                f"get_relation_between_objects: query relation between {a} and {b}"
            ],
            "execution_template": {
                "description": f"Query spatial relation between {a} and {b}",
                "steps": [
                    {
                        "step_id": 0,
                        "function": "get_relation_between_objects",
                        "parameters": {"object_a": a, "object_b": b},
                        "description": f"Find spatial relation from {a} to {b}"
                    }
                ],
                "final_format": "string"
            }
        })

    dir_anchor_names: dict[tuple, list] = defaultdict(list)
    for r in relation_data:
        dir_anchor_names[(r["relation"], r["ObjectB"])].append(r["ObjectA"])

    name_color: dict[str, str | None] = {}
    name_shape: dict[str, str | None] = {}
    COLOR_NAMES = {
        0xff0000: "red", 0x00ff00: "green", 0x0000ff: "blue",
        0xffff00: "yellow", 0x00ffff: "cyan", 0xff00ff: "magenta",
        0xffffff: "white", 0x000000: "black", 0xffa500: "orange",
        0x808080: "gray", 0x800080: "purple", 0x964b00: "brown",
        0xadd8e6: "lightblue", 0x90ee90: "lightgreen", 0xffc0cb: "pink",
    }

    def _cname(raw):
        if raw is None:
            return None
        if isinstance(raw, int):
            iv = raw & 0xFFFFFF
        elif isinstance(raw, str):
            s = raw.strip().lstrip("#")
            if s.startswith("0x"):
                s = s[2:]
            try:
                iv = int(s, 16) & 0xFFFFFF
            except ValueError:
                return None
        else:
            return None
        r2, g2, b2 = (iv >> 16) & 0xFF, (iv >> 8) & 0xFF, iv & 0xFF
        return COLOR_NAMES.get(iv, f"#{r2:02x}{g2:02x}{b2:02x}")

    for obj in objects:
        name_color[obj.get("name")] = _cname(obj.get("color"))
        name_shape[obj.get("name")] = obj.get("type")

    seen_r2: set = set()
    seen_r3: set = set()
    seen_r4: set = set()

    for (direction, anchor), names in dir_anchor_names.items():
        unique_names = list(dict.fromkeys(names))

        key_r2 = (direction, anchor)
        if key_r2 not in seen_r2:
            qa.append({
                "type": "relation",
                "question": f"How many objects are {direction} of {anchor}?",
                "answer": str(len(unique_names)),
                "workflow_steps": [
                    f"count_objects_in_direction: count objects {direction} of {anchor}"
                ],
                "execution_template": {
                    "description": f"Count objects {direction} of {anchor}",
                    "steps": [
                        {
                            "step_id": 0,
                            "function": "count_objects_in_direction",
                            "parameters": {"direction": direction, "anchor_object": anchor},
                            "description": f"Count objects in direction '{direction}' relative to {anchor}"
                        }
                    ],
                    "final_format": "integer_as_string"
                }
            })
            seen_r2.add(key_r2)

        if len(unique_names) >= 1:
            key_r3 = (direction, anchor)
            if key_r3 not in seen_r3:
                colors = []
                for obj_name in unique_names:
                    color = name_color.get(obj_name)
                    if color:
                        colors.append(color)

                if colors:
                    unique_colors = list(dict.fromkeys(colors))

                    if len(unique_names) == 1:
                        qa.append({
                            "type": "relation",
                            "question": f"What color is the object {direction} of {anchor}?",
                            "answer": unique_colors[0],
                            "workflow_steps": [
                                f"get_color_of_object_in_direction: get color of object {direction} of {anchor}"
                            ],
                            "execution_template": {
                                "description": f"Query color of object {direction} of {anchor}",
                                "steps": [
                                    {
                                        "step_id": 0,
                                        "function": "get_color_of_object_in_direction",
                                        "parameters": {"direction": direction, "anchor_object": anchor},
                                        "description": f"Get color of object {direction} of {anchor}"
                                    }
                                ],
                                "final_format": "string"
                            }
                        })
                    else:
                        if len(unique_colors) == 1:
                            qa.append({
                                "type": "relation",
                                "question": f"What color are the objects {direction} of {anchor}?",
                                "answer": unique_colors[0],
                                "workflow_steps": [
                                    f"get_color_of_object_in_direction: get colors of objects {direction} of {anchor}"
                                ],
                                "execution_template": {
                                    "description": f"Query color(s) of objects {direction} of {anchor}",
                                    "steps": [
                                        {
                                            "step_id": 0,
                                            "function": "get_color_of_object_in_direction",
                                            "parameters": {"direction": direction, "anchor_object": anchor},
                                            "description": f"Get color(s) of objects {direction} of {anchor}"
                                        }
                                    ],
                                    "final_format": "string_or_list"
                                }
                            })
                        else:
                            qa.append({
                                "type": "relation",
                                "question": f"What colors are the objects {direction} of {anchor}?",
                                "answer": unique_colors,
                                "workflow_steps": [
                                    f"get_color_of_object_in_direction: get all colors {direction} of {anchor}"
                                ],
                                "execution_template": {
                                    "description": f"Query all colors of objects {direction} of {anchor}",
                                    "steps": [
                                        {
                                            "step_id": 0,
                                            "function": "get_color_of_object_in_direction",
                                            "parameters": {"direction": direction, "anchor_object": anchor},
                                            "description": f"Get colors of objects {direction} of {anchor}"
                                        }
                                    ],
                                    "final_format": "list_of_strings"
                                }
                            })
                    seen_r3.add(key_r3)

        if len(unique_names) >= 1:
            key_r4 = (direction, anchor)
            if key_r4 not in seen_r4:
                shapes = []
                for obj_name in unique_names:
                    shape = name_shape.get(obj_name)
                    if shape:
                        shapes.append(shape)

                if shapes:
                    unique_shapes = list(dict.fromkeys(shapes))

                    if len(unique_names) == 1:
                        qa.append({
                            "type": "relation",
                            "question": f"What shape is the object {direction} of {anchor}?",
                            "answer": unique_shapes[0],
                            "workflow_steps": [
                                f"get_shape_of_object_in_direction: get shape of object {direction} of {anchor}"
                            ],
                            "execution_template": {
                                "description": f"Query shape of object {direction} of {anchor}",
                                "steps": [
                                    {
                                        "step_id": 0,
                                        "function": "get_shape_of_object_in_direction",
                                        "parameters": {"direction": direction, "anchor_object": anchor},
                                        "description": f"Get shape of object {direction} of {anchor}"
                                    }
                                ],
                                "final_format": "string"
                            }
                        })
                    else:
                        if len(unique_shapes) == 1:
                            qa.append({
                                "type": "relation",
                                "question": f"What shape are the objects {direction} of {anchor}?",
                                "answer": unique_shapes[0],
                                "workflow_steps": [
                                    f"get_shape_of_object_in_direction: get shapes of objects {direction} of {anchor}"
                                ],
                                "execution_template": {
                                    "description": f"Query shape(s) of objects {direction} of {anchor}",
                                    "steps": [
                                        {
                                            "step_id": 0,
                                            "function": "get_shape_of_object_in_direction",
                                            "parameters": {"direction": direction, "anchor_object": anchor},
                                            "description": f"Get shape(s) of objects {direction} of {anchor}"
                                        }
                                    ],
                                    "final_format": "string_or_list"
                                }
                            })
                        else:
                            qa.append({
                                "type": "relation",
                                "question": f"What shapes are the objects {direction} of {anchor}?",
                                "answer": unique_shapes,
                                "workflow_steps": [
                                    f"get_shape_of_object_in_direction: get all shapes {direction} of {anchor}"
                                ],
                                "execution_template": {
                                    "description": f"Query all shapes of objects {direction} of {anchor}",
                                    "steps": [
                                        {
                                            "step_id": 0,
                                            "function": "get_shape_of_object_in_direction",
                                            "parameters": {"direction": direction, "anchor_object": anchor},
                                            "description": f"Get shapes of objects {direction} of {anchor}"
                                        }
                                    ],
                                    "final_format": "list_of_strings"
                                }
                            })
                    seen_r4.add(key_r4)

    return qa

def generate_qa(
    objects_path: str,
    color_path: str,
    shape_path: str,
    count_path: str,
    relation_path: str,
    outpath: str,
    start_id: int = 0,
) -> list:
    objects      = _load(objects_path)
    color_data   = _load(color_path)
    shape_data   = _load(shape_path)
    count_data   = _load(count_path)
    relation_data = _load(relation_path)

    all_qa: list = []
    all_qa.extend(_qa_color(objects))
    all_qa.extend(_qa_shape(objects, shape_data))
    all_qa.extend(_qa_count(objects, count_data, shape_data))
    all_qa.extend(_qa_relation(objects, relation_data))

    for i, item in enumerate(all_qa):
        item["id"] = start_id + i

    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        for item in all_qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = len(all_qa)
    by_type = defaultdict(int)
    for item in all_qa:
        by_type[item["type"]] += 1
    print(f"Generated {total} QA items → {outpath}")
    for t, c in sorted(by_type.items()):
        print(f"  [{t}] {c}")

    return all_qa

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate QA templates with execution plans for workflow planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--objects",  required=True, help="Path to semantic_objects.json")
    parser.add_argument("--color",    required=True, help="Path to color.json")
    parser.add_argument("--shape",    required=True, help="Path to shape.json")
    parser.add_argument("--count",    required=True, help="Path to count.json")
    parser.add_argument("--relation", required=True, help="Path to relation.json")
    parser.add_argument("--out",      required=True, help="Output JSONL path")
    parser.add_argument("--start_id", type=int, default=0,
                        help="Starting QA id (default: 0)")
    return parser

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    generate_qa(
        objects_path=args.objects,
        color_path=args.color,
        shape_path=args.shape,
        count_path=args.count,
        relation_path=args.relation,
        outpath=args.out,
        start_id=args.start_id,
    )

