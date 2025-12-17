# AgentsForAgents

A lightweight Poetry-managed Python scaffold for building agentic workflows. The repo ships with a `src`/`tests` layout, Ruff + Black for linting/formatting, MyPy for typing, and GitHub Actions CI.

## Getting started

1. [Install Poetry](https://python-poetry.org/docs/#installation) if it is not already available.
2. Install dependencies:
   ```bash
   poetry install
   ```
3. Run the automated checks:
   ```bash
   ./tools/run_checks.sh
   ```

## Project structure

```
├── configs/          # Environment/configuration templates
├── src/              # Library code (packaged via `src` layout)
├── tests/            # Pytest suite
├── tools/            # Developer tooling helpers (shell scripts, etc.)
└── .github/workflows # CI pipelines
```

## Common commands

| Task | Command |
| --- | --- |
| Format code | `poetry run black .` |
| Lint code | `poetry run ruff check .` |
| Type-check | `poetry run mypy src` |
| Run tests | `poetry run pytest` |

## Using the library

- `agents.core.build_agent_response(messages)` – utility for stitching together agent message history.
- `tools` – retrieval/generation/rerank/evaluation helpers for RAG workflows.
- `agents.PlanningAgent` – delegates planning to GPT-5 (or another configured provider) and maps tool capabilities to an actionable plan or clarifying question.
- `agents.PipelineAgent` – replays saved manifests and surfaces each manifest-defined tool step (plus any custom tool definitions) so you can inspect or implement the resulting pipeline artifacts.

### Example: Planning an agent from a user message

```python
from agents import PlanningAgent

agent = PlanningAgent()
result = agent.plan("Build me a RAG bot that retrieves docs, reranks them, and evaluates answers.")

if result.plan:
    print("Plan:")
    for step in result.plan.steps:
        print(f"- Tool: {step.tool} | Rationale: {step.rationale} | Metadata: {step.metadata}")
else:
    print("Need clarification:", result.clarifying_question)
```

`PlanningAgent` sends the user request plus the available tool inventory to the configured LLM provider (API keys loaded automatically from `.env` via `python-dotenv`) and expects JSON shaped as:

```json
{
  "plan": [{"tool": "...", "rationale": "...", "metadata": {...}}],
  "clarifying_questions": "... or null ...",
  "custom_tools": [
    {
      "name": "CustomFetcher",
      "purpose": "Plain-language summary of why it exists",
      "inputs": "What the tool expects from the user or pipeline (string or JSON)",
      "data_sources": "APIs / DBs / files it connects to (string or JSON)",
      "credentials": "API keys or auth requirements (string or JSON)",
      "metadata": {"linked_plan_step": "Step that needs this tool"}
    }
  ]
}
```

If the built-in registry cannot satisfy a requirement, the planner adds entries to `custom_tools`, asks for the missing non-technical details (inputs, data sources, credentials), and links the finished tool definition to the manifest entry once the plan is finalized.

### Chatting with the planner interactively

Launch the CLI chat loop to iteratively refine requests while preserving conversation history:

```bash
python chat_ui.py --system "You are a pragmatic planner."
```

Type your prompts, review returned plan steps, and enter `exit` when you're done.

While chatting, the planner first checks whether the existing registry covers the user’s needs. When it doesn’t, it will:

1. Explain in plain language why a new tool is required.
2. Ask follow-up questions (inputs, data sources, credentials/API keys) using non-technical wording.
3. Capture the finalized tool definition in the `custom_tools` section, linked to the manifest entry for traceability.

### Materializing and chatting with pipelines

Turn a saved plan into a runnable workspace (pipeline definition, custom tool specs, optional outputs) with:

```bash
python build_pipeline.py <plan_id> --manifest plan_manifests.jsonl
```

This writes files under `pipelines/<plan_id>/`, including `pipeline.json`, `custom_tools/*.json`, and (optionally) `outputs.json` containing the first execution’s context.

End-to-end workflow:

1. **Generate a plan** – Run `python chat_ui.py --mode planner` (or call `PlanningAgent.plan(...)` directly). When the planner finishes without clarifying questions it writes an entry to `plan_manifests.jsonl` that includes the plan steps and any `custom_tools`.
2. **Inspect the manifest** – Each line in `plan_manifests.jsonl` is a JSON object. Locate the `plan_id` you want to deploy (e.g., with `rg <plan_id> plan_manifests.jsonl` or by opening the file in your editor).
3. **Materialize the pipeline** – Execute `python build_pipeline.py <plan_id> --manifest plan_manifests.jsonl`. This snapshots the manifest into `pipelines/<plan_id>/pipeline.json`, writes every custom tool definition to `pipelines/<plan_id>/custom_tools/*.json`, and optionally runs the pipeline once to populate `outputs.json`.
4. **Review the workspace artifacts** – Inspect `pipelines/<plan_id>/pipeline.json` (full manifest snapshot), every file under `pipelines/<plan_id>/custom_tools/` (one JSON per tool definition), and the optional `pipelines/<plan_id>/outputs.json` (initial execution context). Use these files to verify intent before handing the workspace to downstream teams.
5. **Chat with the live pipeline** – Run `python chat_ui.py --mode pipeline --plan-id <plan_id>` (use `--manifest`/`--pipelines-root` if you stored things elsewhere). Each user question becomes the pipeline’s `query`, and responses are streamed back while a chat transcript is appended to `pipelines/<plan_id>/pipeline_chat.jsonl`.

This flow keeps the manifest as the single source of truth, guarantees that every tool definition used by the pipeline is persisted alongside the plan, and gives non-technical stakeholders a simple login-like experience for interacting with the deployed pipeline.

### Configuring providers & manifests

- Install dependencies (`poetry install`) so `openai`, `anthropic`, `python-dotenv`, `pydantic`, and `requests` are available.
- Copy `.env.example` to `.env`, set `LLM_PROVIDER` to your preferred backend, and populate the corresponding credentials. Supported providers:
  - `openai` (default): requires `OPENAI_API_KEY` (optionally `OPENAI_MODEL`).
  - `azure_openai`: requires `AZURE_OPENAI_{API_KEY,ENDPOINT,DEPLOYMENT}` (version optional).
  - `azure_foundry`: requires `AZURE_FOUNDRY_{API_KEY,ENDPOINT,DEPLOYMENT}` (version optional).
  - `claude`: requires `ANTHROPIC_API_KEY` (optionally `CLAUDE_MODEL`).
- The planner calls `load_dotenv()` on import so values from `.env` are available automatically.
- Set `PLAN_MANIFEST_PATH` if you want to store plan manifests somewhere other than `plan_manifests.jsonl`.
- To customize the LLM behavior (e.g., use a mock in tests or a different foundation model), provide a `planner_backend` that implements the `LLMPlanner` protocol:

```python
from agents import PlanningAgent, ToolRegistry
from agents.llm import LLMGeneratedPlan, LLMPlanner, PlanStep

class StubPlanner(LLMPlanner):
    def generate(
        self,
        *,
        user_message: str | None = None,
        registry: ToolRegistry,
        system_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> LLMGeneratedPlan:
        return LLMGeneratedPlan(
            steps=[PlanStep(tool="Mock", rationale=f"prompt={system_prompt}", metadata={})],
        )

agent = PlanningAgent(planner_backend=StubPlanner())
```

To steer the provider's behaviour further, pass a custom `system_prompt` when instantiating `PlanningAgent` (or set it on your custom planner):

```python
custom_prompt = "You are a terse architect. Respond with exactly three steps."
agent = PlanningAgent(system_prompt=custom_prompt)
```

### Executing manifests via `PipelineAgent`

Once a plan is saved to the manifest, you can replay it (manifest-only mode by default):

```python
from agents import PipelineAgent

pipeline = PipelineAgent(manifest_path="plan_manifests.jsonl")
result = pipeline.execute(plan_id="your-plan-uuid")

# Each step writes a payload keyed by its configured `output` metadata (or `<tool>_result`)
for key, payload in result.context.items():
    print(key, payload["tool"], payload.get("status"))
```

Instead of relying on hard-coded tool implementations, `PipelineAgent` now materializes whatever the manifest describes:

- If the plan defines a tool under `custom_tools`, that definition is surfaced verbatim and paired with the step’s execution metadata.
- Otherwise, a generic payload is emitted for the step (`{"tool": ..., "rationale": ..., "metadata": ..., "status": "manifest_defined"}`) so you can inspect the requirements, implement the tool externally, or hand it off to another system.

If you want to execute real tooling, provide your own resolver that knows how to interpret those manifest payloads (or extend `DefaultToolResolver` with concrete handlers for the tool names you control).

## Next steps

- Add your application-specific modules under `src/agents`.
- Duplicate `configs/config.example.yaml` to match your deployment environments.
- Extend the CI workflow with deploy/publish jobs when ready.
- Use the RAG helpers under `src/tools` (retriever, generator, reranker, ragas evaluator) to compose retrieval-augmented agents.
- Leverage `agents.PlanningAgent` to analyze user intents, map available tooling, and produce executable agent plans (or clarifying questions when capabilities are missing).
