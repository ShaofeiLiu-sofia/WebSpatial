from typing import List, Dict, Any, Optional, Union

def get_object_color(object_name: str, semantic_objects: List[Dict[str, Any]]) -> Optional[str]:
    for obj in semantic_objects:
        if obj.get("name") == object_name:
            color = obj.get("color")
            if color is not None:
                return str(color) if isinstance(color, int) else color
    return None

def hex_to_color_name(hex_value: Union[str, int]) -> Optional[str]:
    COLOR_NAMES = {
        0xff0000: "red", 0x00ff00: "green", 0x0000ff: "blue",
        0xffff00: "yellow", 0x00ffff: "cyan", 0xff00ff: "magenta",
        0xffffff: "white", 0x000000: "black", 0xffa500: "orange",
        0x808080: "gray", 0x800080: "purple", 0x964b00: "brown",
        0xadd8e6: "lightblue", 0x90ee90: "lightgreen", 0xffc0cb: "pink",
    }

    try:
        if isinstance(hex_value, str):
            s = hex_value.strip().lstrip("#").lstrip("0x")
            iv = int(s, 16) & 0xFFFFFF
        else:
            iv = int(hex_value) & 0xFFFFFF

        return COLOR_NAMES.get(iv, f"#{iv:06x}")
    except:
        return None

def get_objects_by_color(color_name: str, semantic_objects: List[Dict[str, Any]]) -> List[str]:
    COLOR_TO_HEX = {
        "red": 0xff0000, "green": 0x00ff00, "blue": 0x0000ff,
        "yellow": 0xffff00, "cyan": 0x00ffff, "magenta": 0xff00ff,
        "white": 0xffffff, "black": 0x000000, "orange": 0xffa500,
        "gray": 0x808080, "purple": 0x800080, "brown": 0x964b00,
        "lightblue": 0xadd8e6, "lightgreen": 0x90ee90, "pink": 0xffc0cb,
    }

    target_hex = COLOR_TO_HEX.get(color_name.lower())
    if target_hex is None:
        return []

    result = []
    for obj in semantic_objects:
        obj_color = obj.get("color")
        try:
            if isinstance(obj_color, str):
                s = obj_color.strip().lstrip("#").lstrip("0x")
                obj_hex = int(s, 16) & 0xFFFFFF
            else:
                obj_hex = int(obj_color) & 0xFFFFFF

            if obj_hex == target_hex:
                result.append(obj.get("name"))
        except:
            continue

    return result

def get_object_shape(object_name: str, semantic_objects: List[Dict[str, Any]]) -> Optional[str]:
    for obj in semantic_objects:
        if obj.get("name") == object_name:
            return obj.get("type")
    return None

def get_objects_by_shape(shape_type: str, semantic_objects: List[Dict[str, Any]]) -> List[str]:
    result = []
    for obj in semantic_objects:
        if obj.get("type") == shape_type:
            result.append(obj.get("name"))
    return result

def count_by_shape(shape_type: str, semantic_objects: List[Dict[str, Any]]) -> int:
    return len(get_objects_by_shape(shape_type, semantic_objects))

def count_total_objects(semantic_objects: List[Dict[str, Any]]) -> int:
    return len(semantic_objects)

def count_objects_by_color(color_name: str, semantic_objects: List[Dict[str, Any]]) -> int:
    return len(get_objects_by_color(color_name, semantic_objects))

def count_unique_types(semantic_objects: List[Dict[str, Any]]) -> int:
    types = set(obj.get("type") for obj in semantic_objects if obj.get("type"))
    return len(types)

def count_objects_by_color_and_shape(color_name: str, shape_type: str, semantic_objects: List[Dict[str, Any]]) -> int:
    COLOR_TO_HEX = {
        "red": 0xff0000, "green": 0x00ff00, "blue": 0x0000ff,
        "yellow": 0xffff00, "cyan": 0x00ffff, "magenta": 0xff00ff,
        "white": 0xffffff, "black": 0x000000, "orange": 0xffa500,
        "gray": 0x808080, "purple": 0x800080, "brown": 0x964b00,
        "lightblue": 0xadd8e6, "lightgreen": 0x90ee90, "pink": 0xffc0cb,
    }

    target_hex = COLOR_TO_HEX.get(color_name.lower())
    if target_hex is None:
        return 0

    count = 0
    for obj in semantic_objects:
        if obj.get("type") == shape_type:
            obj_color = obj.get("color")
            try:
                if isinstance(obj_color, str):
                    s = obj_color.strip().lstrip("#").lstrip("0x")
                    obj_hex = int(s, 16) & 0xFFFFFF
                else:
                    obj_hex = int(obj_color) & 0xFFFFFF

                if obj_hex == target_hex:
                    count += 1
            except:
                continue

    return count

def count_objects_not_color(color_name: str, semantic_objects: List[Dict[str, Any]]) -> int:
    total = len(semantic_objects)
    by_color = count_objects_by_color(color_name, semantic_objects)
    return total - by_color

def get_relation_between_objects(object_a: str, object_b: str, relation_data: List[Dict[str, Any]]) -> Optional[str]:
    relations = []
    for r in relation_data:
        if r.get("ObjectA") == object_a and r.get("ObjectB") == object_b:
            relations.append(r.get("relation"))

    if relations:
        unique_relations = list(dict.fromkeys(relations))
        return ", ".join(unique_relations)
    return None

def get_objects_in_direction(direction: str, anchor_object: str, relation_data: List[Dict[str, Any]]) -> List[str]:
    result = []
    for r in relation_data:
        if r.get("relation") == direction and r.get("ObjectB") == anchor_object:
            result.append(r.get("ObjectA"))

    return list(dict.fromkeys(result))

def count_objects_in_direction(direction: str, anchor_object: str, relation_data: List[Dict[str, Any]]) -> int:
    return len(get_objects_in_direction(direction, anchor_object, relation_data))

def get_color_of_object_in_direction(direction: str, anchor_object: str, relation_data: List[Dict[str, Any]], semantic_objects: List[Dict[str, Any]]) -> Union[str, List[str], None]:
    objects_in_dir = get_objects_in_direction(direction, anchor_object, relation_data)
    if not objects_in_dir:
        return None

    colors = []
    for obj_name in objects_in_dir:
        hex_color = get_object_color(obj_name, semantic_objects)
        if hex_color:
            color_name = hex_to_color_name(hex_color)
            if color_name:
                colors.append(color_name)

    if not colors:
        return None
    elif len(objects_in_dir) == 1:
        return colors[0]
    else:
        unique_colors = list(dict.fromkeys(colors))
        return unique_colors[0] if len(unique_colors) == 1 else unique_colors

def get_shape_of_object_in_direction(direction: str, anchor_object: str, relation_data: List[Dict[str, Any]], semantic_objects: List[Dict[str, Any]]) -> Union[str, List[str], None]:
    objects_in_dir = get_objects_in_direction(direction, anchor_object, relation_data)
    if not objects_in_dir:
        return None

    shapes = []
    for obj_name in objects_in_dir:
        shape = get_object_shape(obj_name, semantic_objects)
        if shape:
            shapes.append(shape)

    if not shapes:
        return None
    elif len(objects_in_dir) == 1:
        return shapes[0]
    else:
        unique_shapes = list(dict.fromkeys(shapes))
        return unique_shapes[0] if len(unique_shapes) == 1 else unique_shapes

