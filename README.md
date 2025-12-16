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

### Example: Planning an agent from a user message

```python
from agents import PlanningAgent

agent = PlanningAgent()
result = agent.plan("Build me a RAG bot that retrieves docs, reranks them, and evaluates answers.")

if result.plan:
    print("Plan:")
    for step in result.plan.steps:
        print("-", step)
else:
    print("Need clarification:", result.clarifying_question)
```

`PlanningAgent` sends the user request plus the available tool inventory to OpenAI's GPT-5 models (API key loaded automatically from `.env` via `python-dotenv`) and translates the JSON response into either a plan (steps + rationale) or a clarifying question.

### Extending the tool inventory

You can register custom tools (or override the defaults) and feed them into the planner:

```python
from agents import PlanningAgent, ToolRegistry

registry = ToolRegistry()
registry.register(name="VectorStoreRetriever", category="retrieval", description="Pinecone-backed recall")
registry.register(name="OpenAIGenerator", category="generation", description="GPT-powered responses")

agent = PlanningAgent(registry=registry)
print(agent.plan("Retrieve company policies and answer questions."))
```

Any capabilities you register become eligible for selection when the agent synthesizes plans; missing categories will still trigger clarifying questions so you can iteratively enrich the registry.

Steps to add a custom tool:
1. `from agents import ToolRegistry, PlanningAgent`.
2. Instantiate a registry (or start from `DEFAULT_TOOL_REGISTRY`).
3. Call `registry.register(name=..., category=..., description=...)` for each capability you provide.
4. Pass the registry into `PlanningAgent(registry=registry)` (or `ToolInventory.from_registry(registry)` if you need a reusable snapshot).

The default capabilities are described in `agents/default_tools.json`. You can edit this file, or create your own JSON specification and load it via `ToolRegistry.from_json("path/to/tools.json")` to bootstrap the registry in bulk.

### Configuring OpenAI access

- Install dependencies (`poetry install`) so `openai`, `anthropic`, `python-dotenv`, and `requests` are available.
- Copy `.env.example` to `.env`, set `LLM_PROVIDER` to your preferred backend, and populate the corresponding credentials. Supported providers:
  - `openai` (default): requires `OPENAI_API_KEY` (optionally `OPENAI_MODEL`).
  - `azure_openai`: requires `AZURE_OPENAI_{API_KEY,ENDPOINT,DEPLOYMENT}` (version optional).
  - `azure_foundry`: requires `AZURE_FOUNDRY_{API_KEY,ENDPOINT,DEPLOYMENT}` (version optional).
  - `claude`: requires `ANTHROPIC_API_KEY` (optionally `CLAUDE_MODEL`).
- The planner calls `load_dotenv()` on import so values from `.env` are available automatically.
- To customize the LLM behavior (e.g., use a mock in tests or a different foundation model), provide a `planner_backend` that implements the `LLMPlanner` protocol:

```python
from agents import PlanningAgent, ToolRegistry
from agents.llm import LLMGeneratedPlan, LLMPlanner

class StubPlanner(LLMPlanner):
    def generate(
        self,
        *,
        user_message: str,
        registry: ToolRegistry,
        system_prompt: str | None = None,
    ) -> LLMGeneratedPlan:
        return LLMGeneratedPlan(
            steps=["Mock step"],
            rationale=f"Deterministic test planner (prompt={system_prompt})",
        )

agent = PlanningAgent(planner_backend=StubPlanner())
```

To steer the provider's behaviour further, pass a custom `system_prompt` when instantiating `PlanningAgent` (or set it on your custom planner):

```python
custom_prompt = "You are a terse architect. Respond with exactly three steps."
agent = PlanningAgent(system_prompt=custom_prompt)
```

## Next steps

- Add your application-specific modules under `src/agents`.
- Duplicate `configs/config.example.yaml` to match your deployment environments.
- Extend the CI workflow with deploy/publish jobs when ready.
- Use the RAG helpers under `src/tools` (retriever, generator, reranker, ragas evaluator) to compose retrieval-augmented agents.
- Leverage `agents.PlanningAgent` to analyze user intents, map available tooling, and produce executable agent plans (or clarifying questions when capabilities are missing).
