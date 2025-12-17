import json
from importlib import resources
from typing import Any

import pytest

from agents import PlanningAgent, PlanningResult, ToolInventory, ToolRegistry
from agents.llm import CustomToolDefinition, LLMGeneratedPlan, PlanStep


class StubPlanner:
    def __init__(self, plan: LLMGeneratedPlan):
        self.plan = plan
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        user_message: str | None = None,
        registry: ToolRegistry,
        system_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> LLMGeneratedPlan:
        self.calls.append(
            {
                "user_message": user_message,
                "registry": registry,
                "system_prompt": system_prompt,
                "conversation_history": conversation_history,
            }
        )
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
    assert result.plan_id is not None
    assert result.custom_tools == ()


def test_planning_agent_requests_clarification_when_needed(tmp_path):
    stub_plan = LLMGeneratedPlan(steps=[], clarifying_question="Need more info")
    agent = PlanningAgent(planner_backend=StubPlanner(stub_plan), manifest_path=str(tmp_path / "manifest.jsonl"))

    result = agent.plan("Do something vague")

    assert result.plan is None
    assert result.clarifying_question == "Need more info"
    assert result.plan_id is None
    assert result.custom_tools == ()


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
    assert stub.calls
    assert stub.calls[0]["registry"].capabilities()[0].name == "VectorRetriever"
    assert stub.calls[0]["system_prompt"] is None


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

    assert stub.calls[0]["system_prompt"] == "Be concise."


def test_planning_agent_surfaces_conversation_history(tmp_path):
    returned_history = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Need a plan"},
        {"role": "AI", "content": '{"plan": [], "clarifying_questions": null}'},
    ]
    stub_plan = LLMGeneratedPlan(
        steps=[PlanStep(tool="VectorRetriever", rationale="Reason", metadata={})],
        conversation_history=returned_history,
    )
    agent = PlanningAgent(planner_backend=StubPlanner(stub_plan), manifest_path=str(tmp_path / "manifest.jsonl"))

    result = agent.plan("Need a plan")

    assert list(result.conversation_history) == returned_history


def test_planning_agent_infers_user_message_from_history(tmp_path):
    conversation_history = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Earlier question"},
        {"role": "AI", "content": "Some answer"},
        {"role": "user", "content": "Latest question"},
    ]
    stub_plan = LLMGeneratedPlan(
        steps=[PlanStep(tool="VectorRetriever", rationale="Reason", metadata={})],
        conversation_history=conversation_history + [{"role": "AI", "content": "Final answer"}],
    )
    stub = StubPlanner(stub_plan)
    agent = PlanningAgent(planner_backend=stub, manifest_path=str(tmp_path / "manifest.jsonl"))

    agent.plan(conversation_history=conversation_history)

    assert stub.calls[0]["user_message"] == "Latest question"


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
        steps=[PlanStep(tool="VectorRetriever", rationale="Reason", metadata={"foo": "bar"})],
        custom_tools=[
            CustomToolDefinition(
                name="CustomFetcher",
                purpose="Pulls records from the vendor API",
                inputs="Product ID",
                data_sources="Vendor REST API",
                credentials="API key stored in secrets manager",
            )
        ],
    )
    manifest_path = tmp_path / "manifest.jsonl"
    agent = PlanningAgent(planner_backend=StubPlanner(stub_plan), manifest_path=str(manifest_path))

    result = agent.plan("Need a plan")

    entries = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(entries) == 1
    payload = json.loads(entries[0])
    assert payload["user_message"] == "Need a plan"
    assert payload["steps"][0]["tool"] == "VectorRetriever"
    assert payload["plan_id"] == result.plan_id
    assert payload["custom_tools"][0]["name"] == "CustomFetcher"


def test_manifest_not_written_when_clarification_needed(tmp_path):
    stub_plan = LLMGeneratedPlan(
        steps=[PlanStep(tool="VectorRetriever", rationale="Reason", metadata={})],
        clarifying_question="Need more info",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    agent = PlanningAgent(planner_backend=StubPlanner(stub_plan), manifest_path=str(manifest_path))

    result = agent.plan("Need a plan")

    assert result.plan_id is None
    assert not manifest_path.exists() or not manifest_path.read_text(encoding="utf-8").strip()


def test_planning_agent_surfaces_custom_tools(tmp_path):
    custom_tool = CustomToolDefinition(
        name="CustomSummarizer",
        purpose="Summarizes PDFs from supplier portal",
        inputs={"format": "PDF", "fields": ["section", "summary_length"]},
        data_sources={"primary": "Supplier portal API"},
        credentials={"type": "oauth", "scopes": ["read:portal"]},
        metadata={"linked_plan_step": "Summarize supplier policies"},
    )
    stub_plan = LLMGeneratedPlan(
        steps=[
            PlanStep(tool="CustomSummarizer", rationale="Need bespoke summarization", metadata={}),
        ],
        custom_tools=[custom_tool],
    )
    manifest_path = tmp_path / "manifest.jsonl"
    agent = PlanningAgent(planner_backend=StubPlanner(stub_plan), manifest_path=str(manifest_path))

    result = agent.plan("Summarize supplier policies")

    assert result.custom_tools == (custom_tool,)
    payload = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    assert payload["custom_tools"][0]["name"] == "CustomSummarizer"
