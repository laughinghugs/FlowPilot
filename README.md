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

- `agents_for_agents.core.build_agent_response(messages)` – utility for stitching together agent message history.
- `agents_for_agents.tools` – retrieval/generation/rerank/evaluation helpers for RAG workflows.
- `agents_for_agents.agents.PlanningAgent` – inspects user intent, checks available tooling, and returns either a stepwise plan or a clarifying question when capabilities are missing.

### Example: Planning an agent from a user message

```python
from agents_for_agents.agents import PlanningAgent

agent = PlanningAgent()
result = agent.plan("Build me a RAG bot that retrieves docs, reranks them, and evaluates answers.")

if result.plan:
    print("Plan:")
    for step in result.plan.steps:
        print("-", step)
else:
    print("Need clarification:", result.clarifying_question)
```

`PlanningAgent.plan(...)` analyzes the free-form request, matches it against the available tool inventory, and either produces a sequenced plan with rationale or prompts for more details when a capability is missing.

### Extending the tool inventory

You can register custom tools (or override the defaults) and feed them into the planner:

```python
from agents_for_agents.agents import PlanningAgent, ToolRegistry

registry = ToolRegistry()
registry.register(name="VectorStoreRetriever", category="retrieval", description="Pinecone-backed recall")
registry.register(name="OpenAIGenerator", category="generation", description="GPT-powered responses")

agent = PlanningAgent(registry=registry)
print(agent.plan("Retrieve company policies and answer questions."))
```

Any capabilities you register become eligible for selection when the agent synthesizes plans; missing categories will still trigger clarifying questions so you can iteratively enrich the registry.

## Next steps

- Add your application-specific modules under `src/agents_for_agents`.
- Duplicate `configs/config.example.yaml` to match your deployment environments.
- Extend the CI workflow with deploy/publish jobs when ready.
- Use the RAG helpers under `src/agents_for_agents/tools` (retriever, generator, reranker, ragas evaluator) to compose retrieval-augmented agents.
- Leverage `agents_for_agents.agents.PlanningAgent` to analyze user intents, map available tooling, and produce executable agent plans (or clarifying questions when capabilities are missing).
