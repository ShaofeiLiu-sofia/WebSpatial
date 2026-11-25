import json
import os
import argparse
import requests
from typing import Dict, List, Any, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import executors
from model_configs import MODEL_CONFIG

API_KEY = os.getenv("WEBSPATIAL_API_KEY") or MODEL_CONFIG["api_key"]
MODEL = os.getenv("WEBSPATIAL_MODEL") or MODEL_CONFIG["model"]
BASE_URL = (os.getenv("WEBSPATIAL_BASE_URL") or MODEL_CONFIG["base_url"]).rstrip("/")
URL = os.getenv("WEBSPATIAL_CHAT_COMPLETIONS_URL") or MODEL_CONFIG["chat_completions_url"] or (
    f"{BASE_URL}/chat/completions" if BASE_URL else ""
)

def api_headers():
    if not all((API_KEY, MODEL, URL)):
        raise RuntimeError("Set WEBSPATIAL_API_KEY and configure the model name and chat completions URL")
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

EXECUTOR_FUNCTIONS = {
    "get_object_color": executors.get_object_color,
    "hex_to_color_name": executors.hex_to_color_name,
    "get_objects_by_color": executors.get_objects_by_color,
    "get_object_shape": executors.get_object_shape,
    "get_objects_by_shape": executors.get_objects_by_shape,
    "count_by_shape": executors.count_by_shape,
    "count_total_objects": executors.count_total_objects,
    "count_objects_by_color": executors.count_objects_by_color,
    "count_unique_types": executors.count_unique_types,
    "count_objects_by_color_and_shape": executors.count_objects_by_color_and_shape,
    "count_objects_not_color": executors.count_objects_not_color,
    "get_relation_between_objects": executors.get_relation_between_objects,
    "get_objects_in_direction": executors.get_objects_in_direction,
    "count_objects_in_direction": executors.count_objects_in_direction,
    "get_color_of_object_in_direction": executors.get_color_of_object_in_direction,
    "get_shape_of_object_in_direction": executors.get_shape_of_object_in_direction,
}

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def append_template_to_library(template: Dict[str, Any], templates_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(templates_path)), exist_ok=True)
    with open(templates_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(template, ensure_ascii=False) + "\n")

class ExecutionEngine:

    def __init__(self, scene_data: Dict[str, Any]):
        self.scene_data = scene_data
        self.context = {}

    def execute_plan(self, plan: Dict[str, Any]) -> Any:
        self.context = {}
        steps = plan.get("steps", [])

        if not steps:
            return None

        last_result = None
        for step in steps:
            step_id = step.get("step_id", 0)
            function_name = step.get("function")
            parameters = step.get("parameters", {})

            if not function_name:
                continue

            resolved_params = self._substitute_result_references(parameters)

            try:
                result = self._call_executor(function_name, resolved_params)
                self.context[f"result_{step_id}"] = result
                last_result = result
            except Exception as e:
                print(f"Error executing step {step_id} (function: {function_name}): {e}")
                return None

        return last_result

    def _substitute_result_references(self, parameters: Dict) -> Dict:
        resolved = {}
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("<result_"):

                result_id = value[1:-1]
                if result_id in self.context:
                    resolved[key] = self.context[result_id]
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved

    def _call_executor(self, function_name: str, parameters: Dict) -> Any:
        if function_name not in EXECUTOR_FUNCTIONS:
            raise ValueError(f"Unknown executor function: {function_name}")

        func = EXECUTOR_FUNCTIONS[function_name]

        if function_name in ["get_object_color", "get_objects_by_color", "get_object_shape",
                              "get_objects_by_shape", "count_by_shape", "count_total_objects",
                              "count_objects_by_color", "count_unique_types",
                              "count_objects_by_color_and_shape", "count_objects_not_color"]:
            parameters["semantic_objects"] = self.scene_data.get("semantic_objects", [])

        if function_name in ["get_relation_between_objects", "get_objects_in_direction",
                              "count_objects_in_direction"]:
            parameters["relation_data"] = self.scene_data.get("relation_data", [])

        if function_name in ["get_color_of_object_in_direction", "get_shape_of_object_in_direction"]:
            parameters["relation_data"] = self.scene_data.get("relation_data", [])
            parameters["semantic_objects"] = self.scene_data.get("semantic_objects", [])

        return func(**parameters)

def match_question_to_template(question: str, templates: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int]:
    template_list = "\n".join([
        f"  [{i}] {t['question']}"
        for i, t in enumerate(templates)
    ])

    prompt = f"""Task: Match a given question to the most similar template question.

User question: "{question}"

Available templates:
{template_list}

Return ONLY the template index number (e.g., 0, 1, 2, ...).
Do not include any other text or explanation.
If no close match exists, return -1."""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": MODEL_CONFIG["workflow"]["match"]["max_tokens"],
        "temperature": MODEL_CONFIG["workflow"]["match"]["temperature"]
    }

    try:
        response = requests.post(URL, headers=api_headers(), json=payload, timeout=MODEL_CONFIG["timeout"])
        response.raise_for_status()
        result = response.json()
        response_text = result["choices"][0]["message"]["content"].strip()

        try:
            template_idx = int(response_text.split()[0])
            if template_idx == -1:
                return None, -1
            if 0 <= template_idx < len(templates):
                return templates[template_idx], template_idx
            print(f"Warning: Invalid template index {template_idx}")
        except (ValueError, IndexError):
            print(f"Warning: Could not parse index from response: {response_text}")
    except Exception as e:
        print(f"Error matching question: {e}")

    return None, -1

def get_related_template_indices(question: str, templates: List[Dict[str, Any]], top_k: int = 3) -> List[int]:
    template_list = "\n".join([
        f"  [{i}] {t['question']}"
        for i, t in enumerate(templates)
    ])

    prompt = f"""Task: Find the {top_k} most relevant template questions for the given new question.

New question: "{question}"

Available templates:
{template_list}

Return ONLY the template indices as comma-separated numbers (e.g., 2,5,0).
Do not include any explanation.
Order by relevance (most relevant first)."""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": MODEL_CONFIG["workflow"]["related"]["max_tokens"],
        "temperature": MODEL_CONFIG["workflow"]["related"]["temperature"]
    }

    try:
        response = requests.post(URL, headers=api_headers(), json=payload, timeout=MODEL_CONFIG["timeout"])
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()

        indices = []
        for token in content.replace(";", ",").split(","):
            token = token.strip()
            if token.lstrip("-").isdigit():
                idx = int(token)
                if 0 <= idx < len(templates):
                    indices.append(idx)
        return indices[:top_k]
    except Exception as e:
        print(f"Error obtaining related templates: {e}")

    return []

def plan_workflow_with_llm(question: str, related_templates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    reference_text = "\n".join([
        f"\nReference {i}:\n"
        f"  Question: {t['question']}\n"
        f"  Type: {t.get('type')}\n"
        f"  Answer: {json.dumps(t.get('answer'), ensure_ascii=False)}\n"
        f"  Execution template: {json.dumps(t.get('execution_template'), ensure_ascii=False)}"
        for i, t in enumerate(related_templates)
    ])

    available_functions = ", ".join(sorted(EXECUTOR_FUNCTIONS.keys()))

    prompt = f"""Task: Generate a structured execution plan for a new question based on reference templates.

New question: "{question}"

Reference templates (with their execution patterns):
{reference_text}

Available executor functions:
{available_functions}

Please:
1. Analyze why the reference templates are relevant to the new question
2. Identify the question type (color, shape, count, or relation)
3. Design an execution plan following the same pattern as reference templates
4. Each step must call one of the available functions with proper parameters
5. Steps can reference previous results using <result_N> notation

Return ONLY valid JSON (no markdown, no explanation) with this structure:
{{
    "type": "color|shape|count|relation",
    "reasoning": "Brief explanation of the mapping",
    "execution_plan": {{
        "steps": [
            {{"step_id": 0, "function": "function_name", "parameters": {{"param1": "value1"}}}},
            {{"step_id": 1, "function": "function_name", "parameters": {{"param1": "<result_0>"}}}}
        ]
    }}
}}

Respond with ONLY the JSON object."""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": MODEL_CONFIG["workflow"]["plan"]["max_tokens"],
        "temperature": MODEL_CONFIG["workflow"]["plan"]["temperature"]
    }

    try:
        response = requests.post(URL, headers=api_headers(), json=payload, timeout=MODEL_CONFIG["timeout"])
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()

        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end >= 0:
            json_text = content[start:end+1]
            data = json.loads(json_text)

            required_keys = ["type", "reasoning", "execution_plan"]
            if all(key in data for key in required_keys):
                if "steps" in data["execution_plan"]:
                    return data
    except Exception as e:
        print(f"Error planning workflow: {e}")

    return None

def validate_answer(answer: Any) -> bool:
    if answer is None:
        return False
    if isinstance(answer, str):
        text = answer.strip().lower()
        return text not in {"", "unknown", "n/a", "none", "answer not found", "error"}
    if isinstance(answer, (list, dict)):
        return len(answer) > 0
    if isinstance(answer, (int, float)):
        return True
    return False

def build_template_record(question: str, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": workflow_result["type"],
        "question": question,
        "answer": workflow_result.get("answer"),
        "reasoning": workflow_result.get("reasoning"),
        "execution_template": workflow_result.get("execution_plan"),
        "workflow_steps": [f"{s['function']}: Step {s.get('step_id')}" for s in workflow_result.get("execution_plan", {}).get("steps", [])],
        "generated": True,
        "timestamp": __import__("datetime").datetime.now().isoformat()
    }

def process_questions(questions_path: str, templates_path: str, objects_path: str, relations_path: str, output_path: str, update_templates: bool = True):
    questions = load_jsonl(questions_path)
    templates = load_jsonl(templates_path)

    with open(objects_path, "r", encoding="utf-8") as f:
        semantic_objects = json.load(f) if objects_path.endswith('.json') else load_jsonl(objects_path)

    with open(relations_path, "r", encoding="utf-8") as f:
        relation_data = json.load(f)

    scene_data = {
        "semantic_objects": semantic_objects,
        "relation_data": relation_data
    }

    engine = ExecutionEngine(scene_data)

    print(f"Loaded {len(questions)} questions from {questions_path}")
    print(f"Loaded {len(templates)} templates from {templates_path}")
    print(f"Scene data: {len(semantic_objects)} objects, {len(relation_data)} relations")

    qa_results: List[Dict[str, Any]] = []
    added_templates = 0

    for i, q_item in enumerate(questions):
        question = q_item.get("question", "")
        print(f"\n[{i+1}/{len(questions)}] Processing: {question[:70]}...")

        matched_template, template_idx = match_question_to_template(question, templates)
        if matched_template and validate_answer(matched_template.get("answer")):
            print(f"  ✓ Direct match to template [{template_idx}]")
            qa_item = {
                "id": q_item.get("id", i),
                "type": matched_template.get("type", "unknown"),
                "question": question,
                "answer": matched_template.get("answer"),
                "template_id": template_idx,
                "source": "direct_template_match"
            }
            qa_results.append(qa_item)
            continue

        print("  ✗ Direct match failed; planning workflow from related templates...")
        related_indices = get_related_template_indices(question, templates, top_k=3)
        related_templates = [templates[idx] for idx in related_indices] if related_indices else []

        if not related_templates:
            print("  ✗ No related templates available")
            qa_results.append({
                "id": q_item.get("id", i),
                "type": "unknown",
                "question": question,
                "answer": None,
                "source": "failed_no_templates"
            })
            continue

        workflow_result = plan_workflow_with_llm(question, related_templates)
        if not workflow_result:
            print("  ✗ Workflow planning failed")
            qa_results.append({
                "id": q_item.get("id", i),
                "type": "unknown",
                "question": question,
                "answer": None,
                "source": "failed_planning"
            })
            continue

        execution_plan = workflow_result.get("execution_plan")
        if not execution_plan:
            print("  ✗ No execution plan generated")
            qa_results.append({
                "id": q_item.get("id", i),
                "type": "unknown",
                "question": question,
                "answer": None,
                "source": "failed_no_plan"
            })
            continue

        answer = engine.execute_plan(execution_plan)
        if not validate_answer(answer):
            print("  ✗ Workflow execution produced invalid answer")
            qa_results.append({
                "id": q_item.get("id", i),
                "type": "unknown",
                "question": question,
                "answer": None,
                "source": "failed_invalid_answer"
            })
            continue

        print(f"  ✓ Workflow execution succeeded")
        print(f"    Answer: {str(answer)[:60]}")

        qa_item = {
            "id": q_item.get("id", i),
            "type": workflow_result.get("type", "unknown"),
            "question": question,
            "answer": answer,
            "source": "workflow_execution",
            "reasoning": workflow_result.get("reasoning")
        }
        qa_results.append(qa_item)

        if update_templates:
            workflow_result["answer"] = answer
            new_template = build_template_record(question, workflow_result)
            append_template_to_library(new_template, templates_path)
            templates.append(new_template)
            added_templates += 1
            print(f"    New template added to library (total: {len(templates)})")

    save_jsonl(qa_results, output_path)
    print(f"\n" + "="*60)
    print(f"Saved {len(qa_results)} QA results to {output_path}")
    if added_templates:
        print(f"Added {added_templates} new templates to {templates_path}")

    by_source = {}
    for item in qa_results:
        src = item.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1

    print("\nResult breakdown by source:")
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count}")

def main():
    parser = argparse.ArgumentParser(
        description="Match questions to templates and execute structured workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--questions", required=True, help="Path to questions.jsonl")
    parser.add_argument("--templates", required=True, help="Path to templates.jsonl")
    parser.add_argument("--objects", required=True, help="Path to semantic_objects.json")
    parser.add_argument("--relations", required=True, help="Path to relation.json")
    parser.add_argument("--out", required=True, help="Output path for qa.jsonl")
    parser.add_argument("--no_update_templates", action="store_true",
                        help="Do not append new templates to library")

    args = parser.parse_args()
    process_questions(
        args.questions,
        args.templates,
        args.objects,
        args.relations,
        args.out,
        update_templates=not args.no_update_templates
    )

if __name__ == "__main__":
    main()

