# WebSpatial

WebSpatial extracts spatial states from executable web applications and constructs question-answer pairs through spatial operators and executable workflows.

## Pipeline

1. **Runtime evidence**
   - Static: `static/get_scene_info.py` extracts the scene; `static/get_semantic_objects.py` and `static/analyze_scene.py` derive objects, attributes, and spatial relations.
   - Dynamic: `dynamic/env_runtime.py` records observations and actions; `dynamic/universal_raw_state.py` extracts runtime state; `dynamic/canonical_state.py` normalizes states and computes transitions.
2. **Question-answer generation**
   - Static: `static/workflow-based_generation/generate_templates.py` builds task templates; `executors.py` provides spatial operators; `check.py` matches or plans workflows and executes them.
   - Dynamic: `generate_canonical_qa` in `dynamic/canonical_state.py` derives questions and answers from state transitions.
3. **Model evaluation**: `eval/eval.py` evaluates generated questions; `eval/model_configs.py` sets shared API parameters.


## Configuration

Fill `MODEL_CONFIG` in `eval/model_configs.py`. Its connection fields are `api_key`, `model`, `base_url`, and `chat_completions_url`; the latter is the full API endpoint. The remaining top-level fields control evaluation requests. `workflow.match`, `workflow.related`, and `workflow.plan` control static QA planning calls. `WEBSPATIAL_API_KEY`, `WEBSPATIAL_MODEL`, `WEBSPATIAL_BASE_URL`, and `WEBSPATIAL_CHAT_COMPLETIONS_URL` override the connection fields.
