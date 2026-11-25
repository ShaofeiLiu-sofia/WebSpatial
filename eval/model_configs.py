MODEL_CONFIG = {
    "api_key": "",
    "model": "",
    "base_url": "",
    "chat_completions_url": "",
    "temperature": 0,
    "max_tokens": 1024,
    "timeout": 300,
    "retries": 3,
    "retry_backoff_seconds": [5, 15],
    "seed": None,
    "top_p": None,
    "reasoning_effort": None,
    "verbosity": None,
    "workflow": {
        "match": {"temperature": 0.1, "max_tokens": 5},
        "related": {"temperature": 0.1, "max_tokens": 20},
        "plan": {"temperature": 0.3, "max_tokens": 300},
    },
}

DEFAULT_API_PARAMS = {
    key: MODEL_CONFIG[key]
    for key in (
        "temperature",
        "max_tokens",
        "timeout",
        "retries",
        "retry_backoff_seconds",
        "seed",
        "top_p",
        "reasoning_effort",
        "verbosity",
    )
}

