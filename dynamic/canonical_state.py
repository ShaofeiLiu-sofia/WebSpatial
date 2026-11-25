from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CANONICAL_SCHEMA_VERSION = "canonical_runtime_state_v1"

@dataclass
class CanonicalObject:
    id: str
    type: str
    attributes: dict[str, Any] = field(default_factory=dict)

@dataclass
class CanonicalRelation:
    subject: str
    relation: str
    object: str

@dataclass
class CanonicalState:
    schema: str
    objects: list[CanonicalObject]
    relations: list[CanonicalRelation]
    global_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "objects": [
                {"id": obj.id, "type": obj.type, "attributes": obj.attributes}
                for obj in self.objects
            ],
            "relations": [
                {"subject": rel.subject, "relation": rel.relation, "object": rel.object}
                for rel in self.relations
            ],
            "global": self.global_state,
        }

def canonicalize_state(raw_state: Any) -> dict[str, Any]:

    objects: list[CanonicalObject] = []
    relations: list[CanonicalRelation] = []
    global_state: dict[str, Any] = {}
    if not isinstance(raw_state, dict):
        return CanonicalState(CANONICAL_SCHEMA_VERSION, objects, relations, {"raw": raw_state}).to_dict()

    for key, value in raw_state.items():
        if _is_grid(value):
            _add_grid_objects(objects, relations, key, value)
        elif key == "blocks" and isinstance(value, list):
            _add_coordinate_objects(objects, key, value, raw_state)
        elif _is_scalar(value):
            global_state[key] = value
        elif key == "position" and isinstance(value, list):
            global_state[key] = value
        else:
            global_state[key] = value

    _add_spatial_relations(objects, relations)
    return CanonicalState(CANONICAL_SCHEMA_VERSION, objects, relations, global_state).to_dict()

def canonicalize_state_with_adapter(raw_state: Any, adapter_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    if not adapter_spec:
        return canonicalize_state(raw_state)
    objects: list[CanonicalObject] = []
    relations: list[CanonicalRelation] = []
    global_state: dict[str, Any] = {}

    for global_mapping in adapter_spec.get("globals", []):
        name = str(global_mapping.get("name") or global_mapping.get("id") or "")
        source = str(global_mapping.get("source") or "")
        if name and source:
            global_state[name] = _get_path(raw_state, source)

    for mapping in adapter_spec.get("mappings", []):
        source = str(mapping.get("source") or "")
        values = _get_path(raw_state, source)
        for item, context in _iter_mapping_items(values):
            obj_type = str(_format_template(mapping.get("type", "object"), item, context))
            obj_id = str(_format_template(mapping.get("id", f"{obj_type}_{{index}}"), item, context))
            if any(obj.id == obj_id for obj in objects):
                suffix_parts = [str(context.get(key)) for key in ["lane", "stack", "outer_index", "index"] if key in context]
                obj_id = f"{obj_id}_{'_'.join(suffix_parts)}" if suffix_parts else f"{obj_id}_{len(objects)}"
            attrs: dict[str, Any] = {}
            for attr_name, attr_expr in dict(mapping.get("attributes", {})).items():
                attrs[str(attr_name)] = _eval_adapter_expr(attr_expr, item, context, raw_state)
            if "position" not in attrs:
                pos = _infer_position_from_context(item, context)
                if pos is not None:
                    attrs["position"] = pos
            objects.append(CanonicalObject(id=obj_id, type=obj_type, attributes=attrs))

    for relation_mapping in adapter_spec.get("relations", []):
        subject = str(relation_mapping.get("subject") or "")
        relation = str(relation_mapping.get("relation") or "")
        obj = str(relation_mapping.get("object") or "")
        if subject and relation and obj:
            relations.append(CanonicalRelation(subject, relation, obj))

    if not objects:
        return canonicalize_state(raw_state)
    _add_spatial_relations(objects, relations)
    return CanonicalState(CANONICAL_SCHEMA_VERSION, objects, relations, global_state).to_dict()

def canonical_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_objects = {obj["id"]: obj for obj in before.get("objects", [])}
    after_objects = {obj["id"]: obj for obj in after.get("objects", [])}
    before_ids = set(before_objects)
    after_ids = set(after_objects)
    added = [after_objects[obj_id] for obj_id in sorted(after_ids - before_ids)]
    removed = [before_objects[obj_id] for obj_id in sorted(before_ids - after_ids)]
    changed: list[dict[str, Any]] = []

    for obj_id in sorted(before_ids & after_ids):
        before_attrs = dict(before_objects[obj_id].get("attributes", {}))
        after_attrs = dict(after_objects[obj_id].get("attributes", {}))
        attr_changes = {
            key: {"before": before_attrs.get(key), "after": after_attrs.get(key)}
            for key in sorted(set(before_attrs) | set(after_attrs))
            if before_attrs.get(key) != after_attrs.get(key)
        }
        if attr_changes:
            changed.append(
                {
                    "id": obj_id,
                    "type": after_objects[obj_id].get("type") or before_objects[obj_id].get("type"),
                    "attributes": attr_changes,
                    "before_attributes": before_attrs,
                    "after_attributes": after_attrs,
                }
            )

    before_relations = {_relation_key(rel) for rel in before.get("relations", [])}
    after_relations = {_relation_key(rel) for rel in after.get("relations", [])}
    before_global = dict(before.get("global", {}))
    after_global = dict(after.get("global", {}))
    global_changed = {
        key: {"before": before_global.get(key), "after": after_global.get(key)}
        for key in sorted(set(before_global) | set(after_global))
        if before_global.get(key) != after_global.get(key)
    }
    return {
        "schema": "canonical_runtime_diff_v1",
        "objects_added": added,
        "objects_removed": removed,
        "objects_changed": changed,
        "global_changed": global_changed,
        "relations_added": sorted(after_relations - before_relations),
        "relations_removed": sorted(before_relations - after_relations),
    }

def generate_canonical_qa(
    *,
    env: str,
    frame: dict[str, Any],
    image_before: str,
    image_after: str,
    state_before: dict[str, Any],
    state_after: dict[str, Any],
    diff: dict[str, Any],
    action_sequence: list[str],
    start_count: int,
    end_count: int,
    include_action_sequence: bool = True,
    qa_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_base = f"{env}_{frame['scenario_id']}_{start_count:03d}_{end_count:03d}"
    qa_config = qa_config or {}
    exclude_qa_types = set(qa_config.get("exclude_types", []))
    exclude_attributes = set(qa_config.get("exclude_attributes", []))
    if include_action_sequence:
        rows.append(
            _qa_row(
                sample_id=f"{sample_base}_action",
                env=env,
                qa_type="action_sequence",
                question="Which action sequence changed the runtime state from T0 to T1?",
                answer=",".join(action_sequence),
                images=[image_before, image_after],
                frame=frame,
                state_before=state_before,
                state_after=state_after,
                diff=diff,
            )
        )

    for changed in diff.get("objects_changed", []):
        attrs = changed.get("attributes", {})
        object_label = _object_label(changed)
        if "position" in attrs:
            direction = direction_between(attrs["position"].get("before"), attrs["position"].get("after"))
            if direction and "object_movement_direction" not in exclude_qa_types:
                rows.append(
                    _qa_row(
                        sample_id=f"{sample_base}_{changed['id']}_direction",
                        env=env,
                        qa_type="object_movement_direction",
                        question=f"Which direction did {object_label} move from T0 to T1?",
                        answer=direction,
                        images=[image_before, image_after],
                        frame=frame,
                        state_before=state_before,
                        state_after=state_after,
                        diff=diff,
                    )
                )
            if "object_position_change" not in exclude_qa_types:
                rows.append(
                    _qa_row(
                        sample_id=f"{sample_base}_{changed['id']}_position",
                        env=env,
                        qa_type="object_position_change",
                        question=f"What is the position change of {object_label} from T0 to T1?",
                        answer=f"{format_position(attrs['position'].get('before'))}->{format_position(attrs['position'].get('after'))}",
                        images=[image_before, image_after],
                        frame=frame,
                        state_before=state_before,
                        state_after=state_after,
                        diff=diff,
                    )
                )
        for attr_name, values in attrs.items():
            if attr_name == "position" or attr_name in exclude_attributes:
                continue
            if "object_attribute_change" in exclude_qa_types:
                continue
            rows.append(
                _qa_row(
                    sample_id=f"{sample_base}_{changed['id']}_{attr_name}",
                    env=env,
                    qa_type="object_attribute_change",
                    question=f"How did attribute {attr_name} of {object_label} change from T0 to T1?",
                    answer=f"{values.get('before')}->{values.get('after')}",
                    images=[image_before, image_after],
                    frame=frame,
                    state_before=state_before,
                    state_after=state_after,
                    diff=diff,
                )
            )

    countable_types = {
        obj.get("type")
        for obj in state_after.get("objects", [])
        if obj.get("type") and obj.get("type") not in {"cell", "wall"}
    }
    for object_type in sorted(countable_types):
        if "object_count" in exclude_qa_types:
            continue
        count = sum(1 for obj in state_after.get("objects", []) if obj.get("type") == object_type)
        rows.append(
            _qa_row(
                sample_id=f"{sample_base}_count_{object_type}",
                env=env,
                qa_type="object_count",
                question=f"How many objects of type {object_type} are visible at T1?",
                answer=str(count),
                images=[image_before, image_after],
                frame=frame,
                state_before=state_before,
                state_after=state_after,
                diff=diff,
            )
        )

    for query in qa_config.get("attribute_value_questions", []):
        object_type = query.get("object_type")
        attribute = query.get("attribute")
        time = query.get("time", "T1")
        source_state = state_before if time == "T0" else state_after
        source_image = image_before if time == "T0" else image_after
        for obj in source_state.get("objects", []):
            if object_type and obj.get("type") != object_type:
                continue
            attrs = obj.get("attributes", {})
            if attribute not in attrs:
                continue
            object_label = _object_label({"id": obj.get("id"), "type": obj.get("type"), "after_attributes": attrs})
            rows.append(
                _qa_row(
                    sample_id=f"{sample_base}_{obj.get('id')}_{attribute}_value_{time}",
                    env=env,
                    qa_type="object_attribute_value",
                    question=f"What is the {attribute} of {object_label}?",
                    answer=str(attrs.get(attribute)),
                    images=[source_image],
                    frame=frame,
                    state_before=state_before,
                    state_after=state_after,
                    diff=diff,
                    metadata={"attribute": attribute},
                    task_type="color" if attribute == "color" else None,
                    spatial_abilities=["color", "direction"] if attribute == "color" else None,
                )
            )

    for attr_name, values in diff.get("global_changed", {}).items():
        if attr_name in {"clock", "ct", "dt"} or attr_name in exclude_attributes:
            continue
        if "global_attribute_change" in exclude_qa_types:
            continue
        rows.append(
            _qa_row(
                sample_id=f"{sample_base}_global_{attr_name}",
                env=env,
                qa_type="global_attribute_change",
                question=f"How did global runtime attribute {attr_name} change from T0 to T1?",
                answer=f"{values.get('before')}->{values.get('after')}",
                images=[image_before, image_after],
                frame=frame,
                state_before=state_before,
                state_after=state_after,
                diff=diff,
            )
        )

    return rows

def direction_between(before: Any, after: Any) -> str | None:
    if not isinstance(before, list) or not isinstance(after, list) or len(before) < 2 or len(after) < 2:
        return None
    delta0 = after[0] - before[0]
    delta1 = after[1] - before[1]
    if delta0 == -1 and delta1 == 0:
        return "up"
    if delta0 == 1 and delta1 == 0:
        return "down"
    if delta0 == 0 and delta1 == -1:
        return "left"
    if delta0 == 0 and delta1 == 1:
        return "right"
    return None

def _qa_row(
    *,
    sample_id: str,
    env: str,
    qa_type: str,
    question: str,
    answer: str,
    images: list[str],
    frame: dict[str, Any],
    state_before: dict[str, Any],
    state_after: dict[str, Any],
    diff: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    task_type: str | None = None,
    spatial_abilities: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source_env": f"dynamic_{env}",
        "qa_type": qa_type,
        "task_type": task_type or "relation",
        "spatial_abilities": spatial_abilities or ["relation"],
        "question": question,
        "answer": answer,
        "pic": images,
        "input": {"state_before": state_before, "state_after": state_after, "diff": diff},
        "metadata": {
            "scenario_id": frame.get("scenario_id"),
            "seed": frame.get("seed"),
            "capture_mode": frame.get("capture_mode"),
            "canonical_schema": CANONICAL_SCHEMA_VERSION,
            **(metadata or {}),
        },
    }

def _is_grid(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(row, list) for row in value)

def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None

def _add_grid_objects(
    objects: list[CanonicalObject],
    relations: list[CanonicalRelation],
    key: str,
    grid: list[list[Any]],
) -> None:
    for row_idx, row in enumerate(grid):
        for col_idx, cell_value in enumerate(row):
            cell_id = f"{key}_cell_{row_idx}_{col_idx}"
            objects.append(
                CanonicalObject(
                    id=cell_id,
                    type="cell",
                    attributes={"position": [row_idx, col_idx], "value": cell_value, "source": key},
                )
            )
            if _is_empty_cell(cell_value):
                continue
            entity_type = _entity_type_for_cell(cell_value)
            entity_id = f"{entity_type}_{row_idx}_{col_idx}"
            objects.append(
                CanonicalObject(
                    id=entity_id,
                    type=entity_type,
                    attributes={"position": [row_idx, col_idx], "value": cell_value, "source": key},
                )
            )
            relations.append(CanonicalRelation(entity_id, "on", cell_id))

def _add_coordinate_objects(
    objects: list[CanonicalObject],
    key: str,
    blocks: list[Any],
    raw_state: dict[str, Any],
) -> None:
    origin = raw_state.get("position")
    shape = raw_state.get("currType") or raw_state.get("shape")
    if shape:
        objects.append(CanonicalObject(id="active_shape", type="shape", attributes={"shape": shape, "position": origin}))
    for idx, block in enumerate(blocks):
        if isinstance(block, dict):
            position = [block.get(axis) for axis in ["x", "y", "z"] if axis in block]
        elif isinstance(block, list):
            position = block
        else:
            position = None
        attrs = {"position": position, "source": key}
        if shape:
            attrs["shape"] = shape
        objects.append(CanonicalObject(id=f"{key}_{idx}", type="block", attributes=attrs))

def _entity_type_for_cell(value: Any) -> str:
    text = str(value)
    if text.isdigit():
        return "tile"
    if "player" in text:
        return "player"
    if "block" in text or "box" in text:
        return "box"
    if "target" in text or text in {".", "*"}:
        return "target"
    if "wall" in text or text == "#":
        return "wall"
    return "object"

def _is_empty_cell(value: Any) -> bool:
    return value in (None, "", 0, "empty", "floor")

def _add_spatial_relations(objects: list[CanonicalObject], relations: list[CanonicalRelation]) -> None:
    positioned = [
        obj for obj in objects
        if isinstance(obj.attributes.get("position"), list) and len(obj.attributes["position"]) >= 2
    ]
    by_pos = {
        (obj.attributes["position"][0], obj.attributes["position"][1]): obj
        for obj in positioned
        if obj.type != "cell"
    }
    for obj in positioned:
        if obj.type == "cell":
            continue
        row, col = obj.attributes["position"][0], obj.attributes["position"][1]
        for relation, pos in [
            ("above", (row + 1, col)),
            ("below", (row - 1, col)),
            ("left_of", (row, col + 1)),
            ("right_of", (row, col - 1)),
        ]:
            other = by_pos.get(pos)
            if other and other.id != obj.id:
                relations.append(CanonicalRelation(obj.id, relation, other.id))

def _relation_key(rel: dict[str, Any]) -> str:
    return f"{rel.get('subject')}|{rel.get('relation')}|{rel.get('object')}"

def _get_path(root: Any, path: str) -> Any:
    if not path:
        return root
    if path.startswith("$."):
        path = path[2:]
    current = root
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)] if int(part) < len(current) else None
        else:
            return None
    return current

def _iter_mapping_items(values: Any):
    if isinstance(values, list):
        for index, item in enumerate(values):
            if isinstance(item, list):
                for nested_index, nested_item in enumerate(item):
                    yield nested_item, {"index": nested_index, "outer_index": index, "lane": index, "stack": nested_index}
            else:
                yield item, {"index": index}
    elif isinstance(values, dict):
        for index, (key, item) in enumerate(values.items()):
            yield item, {"index": index, "key": key}
    elif values is not None:
        yield values, {"index": 0}

def _eval_adapter_expr(expr: Any, item: Any, context: dict[str, Any], root: Any = None) -> Any:
    if isinstance(expr, dict):
        op = expr.get("op")
        if op == "map":
            value = _eval_adapter_expr(expr.get("source"), item, context, root)
            mapping = dict(expr.get("mapping", {}))
            return mapping.get(str(value), expr.get("default", value))
        if op == "add_mod":
            values = [_eval_adapter_expr(value, item, context, root) for value in expr.get("values", [])]
            mod = int(expr.get("mod", 1) or 1)
            return sum(int(value or 0) for value in values) % mod
        if op == "sub_mod":
            values = [_eval_adapter_expr(value, item, context, root) for value in expr.get("values", [])]
            mod = int(expr.get("mod", 1) or 1)
            if not values:
                return 0
            result = int(values[0] or 0)
            for value in values[1:]:
                result -= int(value or 0)
            return result % mod
        if op == "format":
            return _format_template(str(expr.get("template", "")), item, {**context, **{k: _eval_adapter_expr(v, item, context, root) for k, v in dict(expr.get("values", {})).items()}})
    if isinstance(expr, list):
        return [_eval_adapter_expr(part, item, context, root) for part in expr]
    if not isinstance(expr, str):
        return expr
    if expr in context:
        return context[expr]
    if expr.startswith("$."):
        return _get_path(root, expr)
    if expr.startswith("item."):
        return _get_path(item, expr[len("item."):])
    if expr == "item":
        return item
    return _format_template(expr, item, context) if "{" in expr else _get_path(item, expr)

def _object_label(changed: dict[str, Any]) -> str:
    attrs = changed.get("after_attributes") or changed.get("attributes_after") or {}
    if not attrs:
        attrs = changed.get("current_attributes") or {}
    before_after_attrs = changed.get("attributes", {})
    name_change = before_after_attrs.get("name") or before_after_attrs.get("display_name")
    if isinstance(name_change, dict):
        return str(name_change.get("before") or name_change.get("after"))
    for attr_name in ["name", "display_name", "label"]:
        value = changed.get(attr_name) or attrs.get(attr_name)
        if value:
            return str(value)
    color_change = before_after_attrs.get("color")
    color = color_change.get("before") if isinstance(color_change, dict) else None
    if color:
        return f"the {color} {changed.get('type', 'object')}"
    return f"the {changed.get('type', 'object')} with id {changed.get('id')}"

def format_position(position: Any) -> str:
    if isinstance(position, list):
        if len(position) == 2:
            return f"lane {position[0]}, stack {position[1]}"
        return "(" + ",".join(str(item) for item in position) + ")"
    return str(position)

def _format_template(template: Any, item: Any, context: dict[str, Any]) -> Any:
    if not isinstance(template, str):
        return template
    values = {**context}
    if isinstance(item, dict):
        values.update(item)
    try:
        return template.format(**values)
    except Exception:
        return template

def _infer_position_from_context(item: Any, context: dict[str, Any]) -> list[Any] | None:
    if isinstance(item, dict):
        if "position" in item and isinstance(item["position"], list):
            return item["position"]
        if {"x", "y", "z"}.issubset(item):
            return [item["x"], item["y"], item["z"]]
        if {"row", "col"}.issubset(item):
            return [item["row"], item["col"]]
    if "lane" in context and "stack" in context:
        return [context["lane"], context["stack"]]
    if "index" in context:
        return [context["index"]]
    return None

