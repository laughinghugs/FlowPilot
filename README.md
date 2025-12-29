# FlowPilot CLI Workflow

## Overview & Prerequisites
- Python 3.11+
- [Poetry](https://python-poetry.org/) for dependency and virtualenv management
- valid `.env` with `LLM_PROVIDER` (`openai` or `azure_openai`) plus the corresponding API keys/endpoints
- optional: set `PLAN_MANIFEST_PATH` to customize where manifests are written (defaults to `plan_manifests.jsonl`)

Quick bootstrap:
```bash
poetry install
cp .env.example .env  # then fill in provider credentials
```

## End-to-End CLI Engine
1. **Plan interactively**
   ```bash
   python chat_ui.py --mode planner
   ```
   - converse with the planner until a plan is finalized (no clarifying questions)
   - every run appends a JSON line to `plan_manifests.jsonl` containing the LLM-summarized “user_message”, the plan steps, and any custom tool definitions

2. **Inspect the manifest**
   - open `plan_manifests.jsonl` (or use `rg <plan_id> plan_manifests.jsonl`) to find:
     - `plan_id`: UUID to feed subsequent commands
     - `user_message`: LLM-generated summary of the conversation and requirements
     - `steps` + `custom_tools`: the exact pipeline specification

3. **Materialize the pipeline & codebase**
   ```bash
   python build_pipeline.py <plan_id> --manifest plan_manifests.jsonl
   ```
   - snapshots the manifest into `pipelines/<plan_id>/pipeline.json`
   - dumps each custom tool definition into `pipelines/<plan_id>/custom_tools/*.json`
   - (optional) executes the manifest once and writes `pipelines/<plan_id>/outputs.json`
   - generates a full Python package under `pipelines/<plan_id>/agent_pipeline/`:
     - `tools.py` – auto-generated implementations/stubs for every tool described in the manifest (LLM-powered when credentials are configured; deterministic fallback otherwise)
     - `pipeline.py` – orchestration layer that wires tools according to manifest order/metadata
     - `agent.py` – simple `AgenticPipeline` façade with a `run(query, context=None)` method

4. **Chat with the resulting agent**
   ```bash
   python chat_ui.py --mode pipeline --plan-id <plan_id>
   ```
   - each prompt becomes the pipeline’s `query`
   - responses stream back while a chat transcript is stored at `pipelines/<plan_id>/pipeline_chat.jsonl`

## Workspace Artifacts
After `build_pipeline.py <plan_id>` completes, expect:
- `pipelines/<plan_id>/pipeline.json` – immutable snapshot of the manifest (includes LLM summary, plan steps, custom tools)
- `pipelines/<plan_id>/custom_tools/*.json` – one file per custom tool definition
- `pipelines/<plan_id>/outputs.json` – (optional) first-run context/results
- `pipelines/<plan_id>/agent_pipeline/` – generated Python codebase
  - `tools.py` – tool implementations/stubs that reflect manifest metadata and expected behavior
  - `pipeline.py` – ordered pipeline definition with `run_pipeline(query, context=None)`
  - `agent.py` – `AgenticPipeline` class exposing `run` for downstream apps
  - `__init__.py` – exports the agent package

Use these files as the handoff bundle: operators get the summarized requirements, executable stubs, and transcripts, while engineers can edit the generated code to provide real integrations if needed.
