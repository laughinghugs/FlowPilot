import json
from importlib import resources

import pytest

from agents import PlanningAgent, PlanningResult, ToolInventory, ToolRegistry
from agents.llm import LLMGeneratedPlan, PlanStep


class StubPlanner:
    def __init__(self, plan: LLMGeneratedPlan):
        self.plan = plan
        self.calls: list[tuple[str, ToolRegistry, str | None]] = []

    def generate(
        self,
        *,
        user_message: str,
        registry: ToolRegistry,
        system_prompt: str | None = None,
    ) -> LLMGeneratedPlan:
        self.calls.append((user_message, registry, system_prompt))
        return self.plan


def test_planning_agent_generates_plan_for_rag_request(tmp_path):
    stub_plan = LLMGeneratedPlan(
        steps=[
            PlanStep(tool="InMemoryRetriever", rationale="Recall documents", metadata={"top_k": 3}),
            PlanStep(tool="TemplateLLMGenerator", rationale="Compose answer", metadata={}),
        ]
    )
    manifest_path = tmp_path / "manifest.jsonl"
    agent = PlanningAgent(planner_backend=StubPlanner(stub_plan), manifest_path=str(manifest_path))

    result = agent.plan("Need a RAG workflow to retrieve docs and generate answers")

    assert isinstance(result, PlanningResult)
    assert result.plan is not None
    assert result.clarifying_question is None
    assert len(result.plan.steps) == 2


def test_planning_agent_requests_clarification_when_needed(tmp_path):
    stub_plan = LLMGeneratedPlan(steps=[], clarifying_question="Need more info")
    agent = PlanningAgent(planner_backend=StubPlanner(stub_plan), manifest_path=str(tmp_path / "manifest.jsonl"))

    result = agent.plan("Do something vague")

    assert result.plan is None
    assert result.clarifying_question == "Need more info"


def test_planning_agent_uses_custom_registry_entries(tmp_path):
    registry = ToolRegistry()
    registry.register(name="VectorRetriever", category="retrieval", description="vector search")
    registry.register(name="TemplateLLMGenerator", category="generation", description="template responses")
    stub_plan = LLMGeneratedPlan(
        steps=[
            PlanStep(tool="VectorRetriever", rationale="Search dense vectors", metadata={"top_k": 5}),
            PlanStep(tool="TemplateLLMGenerator", rationale="Format answer", metadata={}),
        ]
    )
    stub = StubPlanner(stub_plan)
    agent = PlanningAgent(registry=registry, planner_backend=stub, manifest_path=str(tmp_path / "manifest.jsonl"))

    result = agent.plan("Retrieve knowledge base entries")

    assert result.plan is not None
    assert any(step.tool == "VectorRetriever" for step in result.plan.steps)
    assert stub.calls and stub.calls[0][1].capabilities()[0].name == "VectorRetriever"
    assert stub.calls[0][2] is None


def test_planning_agent_passes_system_prompt_to_backend(tmp_path):
    stub_plan = LLMGeneratedPlan(
        steps=[PlanStep(tool="VectorRetriever", rationale="Reason", metadata={})]
    )
    stub = StubPlanner(stub_plan)
    agent = PlanningAgent(
        planner_backend=stub,
        system_prompt="Be concise.",
        manifest_path=str(tmp_path / "manifest.jsonl"),
    )

    agent.plan("Retrieve stuff")

    assert stub.calls[0][2] == "Be concise."


def test_default_registry_matches_json_spec():
    resource = resources.files("agents").joinpath("default_tools.json")
    expected = json.loads(resource.read_text(encoding="utf-8"))

    registry = ToolRegistry.with_default_tools()

    assert [cap.name for cap in registry.capabilities()] == [item["name"] for item in expected]


def test_registry_can_load_external_json(tmp_path):
    data = [
        {"name": "VectorRetriever", "category": "retrieval", "description": "vector db"},
        {"name": "LLM", "category": "generation", "description": "llm"},
    ]
    path = tmp_path / "custom_tools.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    registry = ToolRegistry.from_json(str(path))

    assert len(registry.capabilities()) == 2
    assert registry.capabilities()[0].name == "VectorRetriever"
def test_planning_agent_writes_manifest(tmp_path):
    stub_plan = LLMGeneratedPlan(
        steps=[PlanStep(tool="VectorRetriever", rationale="Reason", metadata={"foo": "bar"})]
    )
    manifest_path = tmp_path / "manifest.jsonl"
    agent = PlanningAgent(planner_backend=StubPlanner(stub_plan), manifest_path=str(manifest_path))

    agent.plan("Need a plan")

    entries = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(entries) == 1
    payload = json.loads(entries[0])
    assert payload["user_message"] == "Need a plan"
    assert payload["steps"][0]["tool"] == "VectorRetriever"
