import json
import base64
import re
import os
import argparse
import time
from pathlib import Path

from model_configs import DEFAULT_API_PARAMS, MODEL_CONFIG

DIRECTION_SEQUENCE_TYPE = "direction_sequence"
DIRECTION_VALUES = ["up", "down", "left", "right"]
HEXTRIS_ROTATION_SEQUENCE_TYPE = "hextris_rotation_sequence"
HEXTRIS_ROTATION_VALUES = ["left", "right"]
HEXTRIS_MAX_SEQUENCE_LENGTH = 3
KLOTSKI_QUESTION_TYPE = "klotski_block_movement"
GRID_CELL_PAIR_TYPE = "grid_cell_pair"
SET_CARD_TRIPLET_TYPE = "set_card_triplet"
QUANTITY_COUNT_TYPE = "quantity_count"
TETRIS_CONTROL_ACTION_TYPE = "tetris_control_action"
GENERATIVE_QUESTION_TYPES = {
    DIRECTION_SEQUENCE_TYPE,
    HEXTRIS_ROTATION_SEQUENCE_TYPE,
    QUANTITY_COUNT_TYPE,
}
ATOMIC_MULTI_SELECT_QUESTION_TYPES = {"color_hex_set", "relation_phrase_set", "shape_name_set"}
OPTIONAL_COMPAT_KEYS = ("top_p", "seed", "reasoning_effort", "verbosity")

API_KEY = os.getenv("WEBSPATIAL_API_KEY") or MODEL_CONFIG["api_key"]
MODEL = os.getenv("WEBSPATIAL_MODEL") or MODEL_CONFIG["model"]
BASE_URL = (os.getenv("WEBSPATIAL_BASE_URL") or MODEL_CONFIG["base_url"]).rstrip("/")
URL = os.getenv("WEBSPATIAL_CHAT_COMPLETIONS_URL") or MODEL_CONFIG["chat_completions_url"] or (
    f"{BASE_URL}/chat/completions" if BASE_URL else ""
)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def none_if_default(value):
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    return value

def parse_retry_backoff(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]

def resolve_runtime_config(args):
    api_params = {**DEFAULT_API_PARAMS}

    cli_overrides = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": None if args.no_seed else args.seed,
        "timeout": args.timeout,
        "retries": args.retries,
        "reasoning_effort": args.reasoning_effort,
        "verbosity": args.verbosity,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            api_params[key] = value
    if args.no_seed:
        api_params["seed"] = None
    if args.retry_backoff is not None:
        api_params["retry_backoff_seconds"] = parse_retry_backoff(args.retry_backoff)

    model = args.model or MODEL
    base_url = (args.base_url or BASE_URL).rstrip("/")
    chat_url = args.chat_completions_url or (
        f"{base_url}/chat/completions" if args.base_url else URL
    )
    return {
        "model": model,
        "base_url": base_url,
        "chat_completions_url": chat_url,
        "api_params": api_params,
    }

def build_prompt(sample, num_images=0, candidate_answers=None):
    question = sample["question"]
    options = sample.get("options") or []
    candidate_answers = candidate_answers or []

    if num_images == 1:
        image_desc = "Carefully analyze the provided image to answer the question."
    elif num_images > 1:
        image_desc = (
            f"Carefully analyze all {num_images} provided images to answer the question. "
            f"The images are provided in temporal order as T0 through T{num_images - 1}; "
            "T0 is the first image and the last T index is the last image."
        )
    else:
        image_desc = "If an image is referenced, infer based on visual semantics."

    if options:
        options_text = "\n".join(options)
        prompt = f"""Task:
- Carefully analyze the given question.
- {image_desc}
- Choose the most accurate answer from the options.
- Multiple answers may be correct.
- Select ALL correct options.

Constraints:
1. Only output a single capital letter from [A, B, C, D].
2. Your answer must strictly match the format like: A
3. If multiple answers, output like: ACD
4. No explanation. No extra text.

Question:
{question}

Options:
{options_text}

Final Answer:
"""
    elif sample.get("question_type") == DIRECTION_SEQUENCE_TYPE:
        direction_values = ", ".join(DIRECTION_VALUES)
        sequence_desc = ""
        if num_images > 2:
            adjacent_pairs = ", ".join(
                f"T{idx}->T{idx + 1}" for idx in range(num_images - 1)
            )
            sequence_desc = (
                f"\n- For these {num_images} images, infer one movement between each adjacent pair: "
                f"{adjacent_pairs}."
            )
        prompt = f"""Task:
- Carefully analyze the given question.
- {image_desc}
- Generate a movement sequence using only the allowed direction values.
{sequence_desc}

Constraints:
1. Use only these direction values: {direction_values}
2. Separate steps with commas.
3. Preserve the required order and repeated steps.
4. Output only the sequence text.
5. No explanation. No extra text.

Question:
{question}

Final Answer:
"""
    elif sample.get("question_type") == HEXTRIS_ROTATION_SEQUENCE_TYPE:
        rotation_values = ", ".join(HEXTRIS_ROTATION_VALUES)
        prompt = f"""Task:
- Carefully analyze the given question.
- {image_desc}
- Generate a Hextris rotation sequence using only the allowed rotation values.

Constraints:
1. Use only these rotation values: {rotation_values}
2. Output between 1 and {HEXTRIS_MAX_SEQUENCE_LENGTH} rotation steps.
3. Separate steps with commas.
4. Preserve the required order and repeated steps.
5. Output only the sequence text.
6. No explanation. No extra text.

Question:
{question}

Final Answer:
"""
    elif sample.get("question_type") == QUANTITY_COUNT_TYPE:
        prompt = f"""Task:
- Carefully analyze the given question.
- {image_desc}
- Generate the exact count requested by the question.

Constraints:
1. Output only one non-negative integer.
2. Do not output words, units, punctuation, or explanation.

Question:
{question}

Final Answer:
"""
    elif sample.get("question_type") == KLOTSKI_QUESTION_TYPE and candidate_answers:
        blocks_text = "\n".join(f"- {answer}" for answer in candidate_answers["block"])
        directions_text = "\n".join(f"- {answer}" for answer in candidate_answers["direction"])
        counts_text = "\n".join(f"- {answer}" for answer in candidate_answers["count"])
        prompt = f"""Task:
- Carefully analyze the given question.
- {image_desc}
- Identify the single Klotski block movement from T0 to T1.

Constraints:
1. Output exactly one movement in this format: block,direction,count
2. The block must be copied exactly from the Block candidates.
3. The direction must be copied exactly from the Direction candidates.
4. The count must be copied exactly from the Count candidates.
5. No spaces around commas.
6. No explanation. No extra text.

Question:
{question}

Block candidates:
{blocks_text}

Direction candidates:
{directions_text}

Count candidates:
{counts_text}

Final Answer:
"""
    elif sample.get("question_type") == GRID_CELL_PAIR_TYPE and candidate_answers:
        answers_text = "\n".join(f"- {answer}" for answer in candidate_answers)
        prompt = f"""Task:
- Carefully analyze the given question.
- {image_desc}
- Identify the two grid cells requested by the question.

Constraints:
1. Each cell in your final answer must be copied exactly from the candidate answer set.
2. Output exactly two cells separated by one comma.
3. Use this format: cell1,cell2
4. No spaces around the comma.
5. No explanation. No extra text.

Question:
{question}

Grid cell candidate set:
{answers_text}

Final Answer:
"""
    elif sample.get("question_type") == SET_CARD_TRIPLET_TYPE and candidate_answers:
        answers_text = "\n".join(f"- {answer}" for answer in candidate_answers)
        prompt = f"""Task:
- Carefully analyze the given question.
- {image_desc}
- Identify the three SET cards requested by the question.

Constraints:
1. Each card in your final answer must be copied exactly from the candidate answer set.
2. Output exactly three cards separated by commas.
3. Use this format: card1,card2,card3
4. No spaces around commas.
5. No explanation. No extra text.

Question:
{question}

SET card candidate set:
{answers_text}

Final Answer:
"""
    elif candidate_answers:
        answers_text = "\n".join(f"- {answer}" for answer in candidate_answers)
        if sample.get("question_type") == TETRIS_CONTROL_ACTION_TYPE:
            prompt = f"""Task:
- Carefully analyze the given question.
- {image_desc}
- Choose the single Tetris control action that best matches the spatial change.

Constraints:
1. Your final answer must be copied exactly from the candidate answer set.
2. Output only one control action.
3. No explanation. No extra text.

Question:
{question}

Candidate answer set:
{answers_text}

Final Answer:
"""
        elif sample.get("question_type") in ATOMIC_MULTI_SELECT_QUESTION_TYPES:
            item_names = {
                "color_hex_set": "color values",
                "relation_phrase_set": "spatial relation phrases",
                "shape_name_set": "answers",
            }
            no_match_names = {
                "color_hex_set": "color value",
                "relation_phrase_set": "spatial relation phrase",
                "shape_name_set": "candidate",
            }
            item_name = item_names[sample.get("question_type")]
            no_match_name = no_match_names[sample.get("question_type")]
            prompt = f"""Task:
- Carefully analyze the given question.
- {image_desc}
- Choose all matching {item_name} from the candidate answer set for this question type.

Constraints:
1. Every item in your final answer must be copied exactly from the candidate answer set.
2. If multiple items are correct, separate them with commas.
3. If there is no matching {no_match_name}, output none.
4. Output only the answer text.
5. No explanation. No extra text.

Question:
{question}

Candidate answer set:
{answers_text}

Final Answer:
"""
        else:
            prompt = f"""Task:
- Carefully analyze the given question.
- {image_desc}
- Choose exactly one answer from the candidate answer set for this question type.

Constraints:
1. Your final answer must be copied exactly from the candidate answer set.
2. Output only the answer text.
3. No explanation. No extra text.

Question:
{question}

Candidate answer set:
{answers_text}

Final Answer:
"""
    else:
        prompt = f"""Task:
- Carefully analyze the given question.
- {image_desc}
- Answer with the shortest exact answer possible.
- No explanation. No extra text.

Question:
{question}

Final Answer:
"""
    return prompt.strip()

def encode_image_to_base64(image_path):
    try:
        image_path = image_path.strip()

        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            return None

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def get_image_mime_type(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp'
    }
    return mime_types.get(ext, 'image/jpeg')

def parse_pic_field(pic_value):
    if not pic_value:
        return []
    if isinstance(pic_value, list):
        return [p for p in pic_value if p]
    if isinstance(pic_value, str):
        return [p.strip() for p in pic_value.split(",") if p.strip()]

    return []

def normalize_text_answer(value):
    if isinstance(value, list):
        return sorted(normalize_text_answer(v) for v in value)
    return str(value).strip().lower().rstrip(".")

def sequence_items(value):
    return [part.strip().lower().rstrip(".") for part in str(value).split(",") if part.strip()]

def is_valid_sequence_from_candidates(value, candidate_answers):
    items = sequence_items(value)
    allowed = set(candidate_answers)
    return bool(items) and all(item in allowed for item in items)

def hextris_net_rotation(value):
    steps = {"left": 1, "right": -1}
    items = sequence_items(value)
    if (
        not items
        or len(items) > HEXTRIS_MAX_SEQUENCE_LENGTH
        or any(item not in steps for item in items)
    ):
        return None
    return sum(steps[item] for item in items) % 6

def is_hextris_net_rotation_correct(prediction, canonical_answer):
    prediction_rotation = hextris_net_rotation(prediction)
    return (
        prediction_rotation is not None
        and prediction_rotation == hextris_net_rotation(canonical_answer)
    )

def parse_klotski_answer(value):
    parts = [part.strip().lower().rstrip(".") for part in str(value).split(",")]
    if len(parts) != 3:
        return None
    block, direction, count = parts
    if "@" not in block or not count.isdigit():
        return None
    return {"block": block, "direction": direction, "count": count}

def is_valid_klotski_answer(value, candidate_answers):
    parsed = parse_klotski_answer(value)
    if parsed is None or not isinstance(candidate_answers, dict):
        return False
    return all(
        parsed[name] in set(candidate_answers.get(name, []))
        for name in ("block", "direction", "count")
    )

def is_valid_set_from_candidates(value, candidate_answers):
    items = sequence_items(value)
    allowed = set(candidate_answers)
    if items == ["none"]:
        return "none" in allowed
    return bool(items) and "none" not in items and all(item in allowed for item in items)

def is_valid_grid_cell_pair(value, candidate_answers):
    items = sequence_items(value)
    allowed = set(candidate_answers)
    return len(items) == 2 and all(item in allowed for item in items)

def is_valid_set_card_triplet(value, candidate_answers):
    items = sequence_items(value)
    allowed = set(candidate_answers)
    return len(items) == 3 and len(set(items)) == 3 and all(item in allowed for item in items)

def is_non_negative_integer(value):
    return bool(re.fullmatch(r"\d+", str(value).strip()))

def normalize_set_answer(value):
    parts = [part.strip().lower().rstrip(".") for part in str(value).split(",") if part.strip()]
    return ", ".join(sorted(set(parts)))

def normalize_prediction_for_sample(value, sample):
    if sample.get("question_type") in {
        "color_hex_set",
        "shape_name_set",
        "relation_phrase_set",
        GRID_CELL_PAIR_TYPE,
        SET_CARD_TRIPLET_TYPE,
    }:
        return normalize_set_answer(value)
    return normalize_text_answer(value)

def accepted_answers_for_sample(sample):
    canonical = normalize_prediction_for_sample(
        sample.get("canonical_answer", sample.get("answer")), sample
    )
    raw_answers = sample.get("accepted_answers") or [canonical]
    answers = [normalize_prediction_for_sample(value, sample) for value in raw_answers]
    if canonical not in answers:
        answers.insert(0, canonical)
    return list(dict.fromkeys(answers))

def parse_answer(output, has_options=False):
    output = output.strip()
    if has_options:
        letters = re.findall(r"[ABCD]", output.upper())
        if not letters:
            return []
        return sorted(set(letters))
    return normalize_text_answer(output.splitlines()[0])

def summarize_response(result):
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning_content = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("thinking")
        or ""
    )
    return {
        "id": result.get("id"),
        "created": result.get("created"),
        "model": result.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "content": content[:4000],
        "content_chars": len(content),
        "reasoning_content": str(reasoning_content)[:4000],
        "reasoning_content_chars": len(str(reasoning_content)),
        "usage": result.get("usage"),
    }

def build_request_payload(prompt, image_paths, runtime_config, max_tokens):
    api_params = runtime_config["api_params"]

    content = [{"type": "text", "text": prompt}]

    if image_paths:
        for image_path in image_paths:
            if not image_path:
                continue

            image_path = image_path.strip()

            if image_path.startswith(("http://", "https://")):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_path}
                })
            else:
                base64_image = encode_image_to_base64(image_path)
                if base64_image:
                    mime_type = get_image_mime_type(image_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })

    data = {
        "model": runtime_config["model"],
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": api_params["temperature"],
        "max_tokens": max_tokens,
    }
    for optional_key in OPTIONAL_COMPAT_KEYS:
        value = none_if_default(api_params.get(optional_key))
        if value is not None:
            data[optional_key] = value
    return data

def should_retry_error(status_code):
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}

def post_with_retries(payload, runtime_config):
    import requests

    api_params = runtime_config["api_params"]
    retries = int(api_params.get("retries") or 0)
    backoff = list(api_params.get("retry_backoff_seconds") or [])
    timeout = int(api_params.get("timeout") or 180)
    last_exc = None

    attempt = 0
    while attempt < retries + 1:
        try:
            response = requests.post(
                runtime_config["chat_completions_url"],
                headers=HEADERS,
                json=payload,
                timeout=timeout,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                status_code = response.status_code
                if attempt < retries and should_retry_error(status_code):
                    time.sleep(backoff[min(attempt, len(backoff) - 1)] if backoff else 0)
                    attempt += 1
                    continue
                raise requests.HTTPError(
                    f"{exc}; response={response.text[:1000]}"
                ) from exc
            return response.json()
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(backoff[min(attempt, len(backoff) - 1)] if backoff else 0)
            attempt += 1
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Request failed before receiving a response.")

def call_model(prompt, image_paths=None, has_options=False, runtime_config=None, max_tokens=256):
    runtime_config = runtime_config or {
        "model": MODEL,
        "chat_completions_url": URL,
        "api_params": {**DEFAULT_API_PARAMS},
    }
    payload = build_request_payload(prompt, image_paths or [], runtime_config, max_tokens)
    result = post_with_retries(payload, runtime_config)
    choice = result["choices"][0]
    output = (choice.get("message", {}).get("content") or "").strip()
    parsed_answer = parse_answer(output, has_options=has_options) if output else ([] if has_options else "")
    return {
        "prediction": parsed_answer,
        "raw_output": output,
        "response": summarize_response(result),
        "request_params": {
            key: payload.get(key)
            for key in ("model", "temperature", "max_tokens", "top_p", "seed", "reasoning_effort", "verbosity")
            if key in payload
        },
        "status": "ok" if output else "empty_output",
        "error": None,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--chat-completions-url", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-seed", action="store_true")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--retry-backoff", default=None)
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None)
    parser.add_argument("--verbosity", choices=["low", "medium", "high"], default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--candidate-dir", default="candidate_answers")
    return parser.parse_args()

def resolve_input_file(input_file):
    if os.path.exists(input_file):
        return input_file
    if not os.path.isabs(input_file):
        candidate = Path(__file__).resolve().parent / input_file
        if candidate.exists():
            return str(candidate)
    return input_file

def resolve_eval_path(path_value, base_dir=None):
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    return base / path

def resolved_image_path(image_path):
    path = Path(str(image_path))
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path

def load_candidate_answers(sample, candidate_dir):
    if sample.get("options"):
        return []
    if sample.get("question_type") in GENERATIVE_QUESTION_TYPES:
        return []
    if sample.get("question_type") == KLOTSKI_QUESTION_TYPE:
        candidate_files = sample.get("candidate_answer_files") or {
            "block": "candidate_answers/klotski_block_movement.blocks.json",
            "direction": "candidate_answers/klotski_block_movement.directions.json",
            "count": "candidate_answers/klotski_block_movement.counts.json",
        }
        loaded = {}
        for name in ("block", "direction", "count"):
            path = resolve_eval_path(candidate_files[name])
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Candidate answer file must be a JSON array: {path}")
            normalized = [normalize_text_answer(answer) for answer in data]
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"Candidate answer file contains duplicate answers: {path}")
            loaded[name] = normalized
        return loaded

    candidate_file = sample.get("candidate_answers_file")
    if candidate_file:
        path = resolve_eval_path(candidate_file)
    elif sample.get("question_type"):
        path = resolve_eval_path(candidate_dir) / f"{sample['question_type']}.json"
    else:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Candidate answer file must be a JSON array: {path}")
    answers = data
    if not isinstance(answers, list):
        raise ValueError(f"Candidate answers must be a list: {path}")
    normalized = [normalize_text_answer(answer) for answer in answers]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Candidate answer file contains duplicate answers: {path}")
    return normalized

def check_eval_file(lines, candidate_dir):
    errors = []
    for idx, line in enumerate(lines):
        sample = json.loads(line)
        images = parse_pic_field(sample.get("pic"))
        if not images:
            errors.append(f"line {idx + 1}: no images in pic")
            continue
        for image in images:
            if not resolved_image_path(image).exists():
                errors.append(f"line {idx + 1}: missing image {image}")
                break
        if not sample.get("options"):
            if not sample.get("question_type"):
                errors.append(f"line {idx + 1}: no question_type")
                continue
            gold = normalize_prediction_for_sample(
                sample.get("canonical_answer", sample.get("answer")), sample
            )
            accepted_answers = accepted_answers_for_sample(sample)
            if sample.get("question_type") == DIRECTION_SEQUENCE_TYPE:
                if not is_valid_sequence_from_candidates(gold, DIRECTION_VALUES):
                    errors.append(
                        f"line {idx + 1}: gold sequence contains invalid direction values"
                    )
                continue
            if sample.get("question_type") == HEXTRIS_ROTATION_SEQUENCE_TYPE:
                if not is_valid_sequence_from_candidates(gold, HEXTRIS_ROTATION_VALUES):
                    errors.append(
                        f"line {idx + 1}: gold sequence contains invalid Hextris rotation values"
                    )
                if hextris_net_rotation(gold) is None:
                    errors.append(
                        f"line {idx + 1}: gold Hextris sequence exceeds the supported length"
                    )
                invalid_accepted = [
                    answer
                    for answer in accepted_answers
                    if not is_hextris_net_rotation_correct(answer, gold)
                ]
                if invalid_accepted:
                    errors.append(
                        f"line {idx + 1}: invalid net-rotation-equivalent answers: {invalid_accepted}"
                    )
                continue
            if sample.get("question_type") == QUANTITY_COUNT_TYPE:
                if not is_non_negative_integer(gold):
                    errors.append(f"line {idx + 1}: gold count is not a non-negative integer")
                continue
            if sample.get("question_type") == KLOTSKI_QUESTION_TYPE:
                if not sample.get("candidate_answer_files"):
                    errors.append(f"line {idx + 1}: no candidate_answer_files")
                    continue
                try:
                    candidate_answers = load_candidate_answers(sample, candidate_dir)
                except Exception as exc:
                    errors.append(f"line {idx + 1}: cannot load candidate answers: {exc}")
                    continue
                if not is_valid_klotski_answer(gold, candidate_answers):
                    errors.append(
                        f"line {idx + 1}: gold Klotski answer contains values absent from component candidates"
                    )
                continue
            if sample.get("question_type") == GRID_CELL_PAIR_TYPE:
                if not sample.get("candidate_answers_file"):
                    errors.append(f"line {idx + 1}: no candidate_answers_file")
                    continue
                try:
                    candidate_answers = load_candidate_answers(sample, candidate_dir)
                except Exception as exc:
                    errors.append(f"line {idx + 1}: cannot load candidate answers: {exc}")
                    continue
                if not is_valid_grid_cell_pair(gold, candidate_answers):
                    errors.append(
                        f"line {idx + 1}: gold grid-cell pair contains values absent from candidate answers"
                    )
                invalid_accepted = [
                    answer
                    for answer in accepted_answers
                    if not is_valid_grid_cell_pair(answer, candidate_answers)
                ]
                if invalid_accepted:
                    errors.append(
                        f"line {idx + 1}: invalid accepted grid-cell pair answers: {invalid_accepted}"
                    )
                continue
            if sample.get("question_type") == SET_CARD_TRIPLET_TYPE:
                if not sample.get("candidate_answers_file"):
                    errors.append(f"line {idx + 1}: no candidate_answers_file")
                    continue
                try:
                    candidate_answers = load_candidate_answers(sample, candidate_dir)
                except Exception as exc:
                    errors.append(f"line {idx + 1}: cannot load candidate answers: {exc}")
                    continue
                if not is_valid_set_card_triplet(gold, candidate_answers):
                    errors.append(
                        f"line {idx + 1}: gold SET-card triplet contains values absent from candidate answers"
                    )
                invalid_accepted = [
                    answer
                    for answer in accepted_answers
                    if not is_valid_set_card_triplet(answer, candidate_answers)
                ]
                if invalid_accepted:
                    errors.append(
                        f"line {idx + 1}: invalid accepted SET-card triplet answers: {invalid_accepted}"
                    )
                continue
            if not sample.get("candidate_answers_file"):
                errors.append(f"line {idx + 1}: no candidate_answers_file")
                continue
            try:
                candidate_answers = load_candidate_answers(sample, candidate_dir)
            except Exception as exc:
                errors.append(f"line {idx + 1}: cannot load candidate answers: {exc}")
                continue
            if sample.get("accepted_answers"):
                invalid_accepted = [
                    answer for answer in accepted_answers if answer not in set(candidate_answers)
                ]
                if invalid_accepted:
                    errors.append(
                        f"line {idx + 1}: accepted answers are absent from candidate answers: {invalid_accepted}"
                    )
                    continue
            if sample.get("question_type") in ATOMIC_MULTI_SELECT_QUESTION_TYPES:
                if not is_valid_set_from_candidates(gold, candidate_answers):
                    errors.append(
                        f"line {idx + 1}: gold answer contains values absent from candidate answers"
                    )
            elif gold not in set(candidate_answers):
                errors.append(f"line {idx + 1}: gold answer is absent from candidate answers")
    return errors

def main():
    global MODEL

    args = parse_args()
    runtime_config = resolve_runtime_config(args)
    MODEL = runtime_config["model"]
    input_file = resolve_input_file(args.input)
    results = []

    strict_correct = 0

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if args.check_only:
        errors = check_eval_file(lines, args.candidate_dir)
        if errors:
            print("Eval file check failed:")
            for error in errors[:20]:
                print(f"- {error}")
            raise SystemExit(1)
        print(f"Eval file check passed: {len(lines)} rows")
        return

    if not all((API_KEY, runtime_config["model"], runtime_config["chat_completions_url"])):
        raise RuntimeError("Set WEBSPATIAL_API_KEY and configure the model name and chat completions URL")

    from tqdm import tqdm

    if args.limit > 0:
        lines = lines[:args.limit]

    for idx, line in enumerate(tqdm(lines)):
        sample = json.loads(line)
        image_paths = parse_pic_field(sample.get("pic"))
        has_options = bool(sample.get("options"))
        candidate_answers = [] if has_options else load_candidate_answers(sample, args.candidate_dir)
        prompt = build_prompt(sample, num_images=len(image_paths), candidate_answers=candidate_answers)
        sample_runtime_config = runtime_config

        try:
            model_result = call_model(
                prompt,
                image_paths,
                has_options=has_options,
                runtime_config=sample_runtime_config,
                max_tokens=sample_runtime_config["api_params"]["max_tokens"],
            )
            pred = model_result["prediction"]
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            pred = [] if has_options else ""
            model_result = {
                "prediction": pred,
                "raw_output": "",
                "response": None,
                "request_params": {
                    "model": sample_runtime_config["model"],
                    "max_tokens": sample_runtime_config["api_params"]["max_tokens"],
                    "temperature": sample_runtime_config["api_params"]["temperature"],
                    "seed": sample_runtime_config["api_params"].get("seed"),
                },
                "status": "error",
                "error": str(e),
            }

        gt = sorted(sample["answer"]) if has_options else normalize_prediction_for_sample(
            sample.get("canonical_answer", sample.get("answer")), sample
        )
        accepted_answers = [gt] if has_options else accepted_answers_for_sample(sample)
        valid_prediction = True
        if not has_options:
            pred = normalize_prediction_for_sample(pred, sample)
            if sample.get("question_type") == DIRECTION_SEQUENCE_TYPE:
                valid_prediction = is_valid_sequence_from_candidates(pred, DIRECTION_VALUES)
            elif sample.get("question_type") == HEXTRIS_ROTATION_SEQUENCE_TYPE:
                valid_prediction = hextris_net_rotation(pred) is not None
            elif sample.get("question_type") == QUANTITY_COUNT_TYPE:
                valid_prediction = is_non_negative_integer(pred)
            elif sample.get("question_type") == KLOTSKI_QUESTION_TYPE:
                valid_prediction = is_valid_klotski_answer(pred, candidate_answers)
            elif sample.get("question_type") == GRID_CELL_PAIR_TYPE:
                valid_prediction = is_valid_grid_cell_pair(pred, candidate_answers)
            elif sample.get("question_type") == SET_CARD_TRIPLET_TYPE:
                valid_prediction = is_valid_set_card_triplet(pred, candidate_answers)
            elif sample.get("question_type") in ATOMIC_MULTI_SELECT_QUESTION_TYPES:
                valid_prediction = is_valid_set_from_candidates(pred, candidate_answers)
            else:
                valid_prediction = pred in set(candidate_answers)
        is_canonical_exact_correct = pred == gt
        if not has_options and sample.get("question_type") == HEXTRIS_ROTATION_SEQUENCE_TYPE:
            is_strict_correct = is_hextris_net_rotation_correct(pred, gt)
        else:
            is_strict_correct = pred in accepted_answers
        strict_correct += int(is_strict_correct)

        results.append({
            "id": sample.get("id", idx),
            "model": runtime_config["model"],
            "source_env": sample.get("source_env"),
            "task_type": sample.get("task_type"),
            "question_type": sample.get("question_type"),
            "question": sample["question"],
            "prediction": pred,
            "answer": gt,
            "accepted_answers": accepted_answers,
            "raw_answer": sample.get("answer"),
            "image_count": len(image_paths),
            "valid_prediction": valid_prediction,
            "strict_correct": is_strict_correct,
            "canonical_exact_correct": is_canonical_exact_correct,
            "model_status": model_result["status"],
            "model_error": model_result["error"],
            "raw_output": model_result["raw_output"],
            "response": model_result["response"],
            "request_params": model_result["request_params"],
        })

    accuracy = strict_correct / len(results) if results else 0
    print(f"\nAccuracy: {accuracy:.2%}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

