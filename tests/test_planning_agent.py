import json
from importlib import resources

import pytest

from agents import PlanningAgent, PlanningResult, ToolInventory, ToolRegistry
from agents.llm import LLMGeneratedPlan


class StubPlanner:
    def __init__(self, plan: LLMGeneratedPlan):
        self.plan = plan
        self.calls: list[tuple[str, ToolRegistry]] = []

    def generate(self, *, user_message: str, registry: ToolRegistry) -> LLMGeneratedPlan:
        self.calls.append((user_message, registry))
        return self.plan


def test_planning_agent_generates_plan_for_rag_request():
    stub_plan = LLMGeneratedPlan(
        steps=["Use InMemoryRetriever for recall", "Generate response via TemplateLLMGenerator"],
        rationale="Covers retrieval + generation",
    )
    agent = PlanningAgent(planner_backend=StubPlanner(stub_plan))

    result = agent.plan("Need a RAG workflow to retrieve docs and generate answers")

    assert isinstance(result, PlanningResult)
    assert result.plan is not None
    assert result.clarifying_question is None
    assert len(result.plan.steps) == 2


def test_planning_agent_requests_clarification_when_needed():
    stub_plan = LLMGeneratedPlan(steps=[], rationale="", clarifying_question="Need more info")
    agent = PlanningAgent(planner_backend=StubPlanner(stub_plan))

    result = agent.plan("Do something vague")

    assert result.plan is None
    assert result.clarifying_question == "Need more info"


def test_planning_agent_uses_custom_registry_entries():
    registry = ToolRegistry()
    registry.register(name="VectorRetriever", category="retrieval", description="vector search")
    registry.register(name="TemplateLLMGenerator", category="generation", description="template responses")
    stub_plan = LLMGeneratedPlan(
        steps=["Use VectorRetriever", "Use TemplateLLMGenerator"],
        rationale="Custom stack",
    )
    stub = StubPlanner(stub_plan)
    agent = PlanningAgent(registry=registry, planner_backend=stub)

    result = agent.plan("Retrieve knowledge base entries")

    assert result.plan is not None
    assert any("VectorRetriever" in step for step in result.plan.steps)
    assert stub.calls and stub.calls[0][1].capabilities()[0].name == "VectorRetriever"


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
