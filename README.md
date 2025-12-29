# FlowPilot CLI Workflow

## Overview & Prerequisites
- Python 3.11+
- [Poetry](https://python-poetry.org/) (recommended) for dependency + virtualenv management
- `.env` containing `LLM_PROVIDER` (`openai` or `azure_openai`) and the corresponding API credentials
- optional: `PLAN_MANIFEST_PATH` if you want manifests somewhere other than `plan_manifests.jsonl`

Bootstrap:
```bash
poetry install                      # or your preferred installer
cp .env.example .env                # then fill in provider credentials
```

## CLI Interaction Flow
1. **Plan with the LLM architect**
   ```bash
   python chat_ui.py --mode planner
   ```
   - The CLI shows “Agent is thinking…” while the planner reasons.
   - If information is missing, the agent asks clarifying questions in plain English.
   - Once it can propose a solution it prints a numbered summary of the steps and prompts you to confirm. Only after you accept (`y`) does it snapshot the conversation into `plan_manifests.jsonl`, including an LLM-authored requirement brief and auto-derived custom tool definitions (one per unique plan step). There is no separate tool registry—everything lives inside the manifest.

2. **Inspect the manifest**
   - Open `plan_manifests.jsonl` (or `rg <plan_id> plan_manifests.jsonl`) to review:
     - `plan_id`: UUID for all downstream commands.
     - `user_message`: concise summary of the entire chat.
     - `steps` and `custom_tools`: the exact sequence and capabilities that will be generated.

3. **Materialize the pipeline workspace**
   ```bash
   python build_pipeline.py <plan_id> --manifest plan_manifests.jsonl
   ```
   - Copies the manifest into `pipelines/<plan_id>/pipeline.json`.
   - Dumps each custom tool definition into `pipelines/<plan_id>/custom_tools/*.json`.
   - Optionally executes the plan once (unless `--no-run` is provided) and saves `pipelines/<plan_id>/outputs.json`.
   - Invokes LLM-based or template-based code generation to create a runnable Python package at `pipelines/<plan_id>/agent_pipeline/` (`tools.py`, `pipeline.py`, `agent.py`, `__init__.py`).

4. **Chat with the generated agent**
   ```bash
   python chat_ui.py --mode pipeline --plan-id <plan_id>
   ```
   - Opens a CLI session backed by the generated pipeline. Your prompts feed into the agent; responses are appended to `pipelines/<plan_id>/pipeline_chat.jsonl`.

## Workspace Artifacts
| Path | Description |
| --- | --- |
| `pipelines/<plan_id>/pipeline.json` | Immutable snapshot of the manifest (summary, steps, derived tools). |
| `pipelines/<plan_id>/custom_tools/*.json` | Each manifest-defined tool stored as a standalone JSON description. |
| `pipelines/<plan_id>/outputs.json` | (Optional) Context/results from the initial pipeline execution. |
| `pipelines/<plan_id>/agent_pipeline/tools.py` | Auto-generated tool implementations/stubs based on manifest metadata. |
| `pipelines/<plan_id>/agent_pipeline/pipeline.py` | Ordered execution plan exposing `run_pipeline(query, context=None)`. |
| `pipelines/<plan_id>/agent_pipeline/agent.py` | `AgenticPipeline` facade with a `run` method for downstream apps. |
| `pipelines/<plan_id>/agent_pipeline/__init__.py` | Package entrypoint exporting `AgenticPipeline`. |

This bundle is the complete deliverable: the manifest captures requirements plus tools, the pipeline workspace holds generated code, and the chat UI lets non-technical users interact with either the planner or the built agent without touching any tool registry infrastructure.
