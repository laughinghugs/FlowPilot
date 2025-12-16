import pytest

from agents_for_agents.agents import PlanningAgent, PlanningResult, ToolInventory, ToolRegistry


def test_planning_agent_returns_plan_when_all_capabilities_available():
    agent = PlanningAgent()

    result = agent.plan("Need a RAG workflow to retrieve docs and generate answers")

    assert isinstance(result, PlanningResult)
    assert result.plan is not None
    assert result.clarifying_question is None
    assert any("retrieval" in step.lower() for step in result.plan.steps)


def test_planning_agent_requests_clarification_when_capability_missing():
    registry = ToolRegistry()
    registry.register(name="Generator", category="generation", description="LLM outputs")
    inventory = ToolInventory.from_registry(registry)
    agent = PlanningAgent(inventory=inventory)

    result = agent.plan("The agent must retrieve knowledge base entries")

    assert result.plan is None
    assert result.clarifying_question is not None
    assert "retrieval" in result.clarifying_question


def test_planning_agent_orders_steps_consistently():
    agent = PlanningAgent()
    result = agent.plan("Please evaluate responses and rerank documents")

    assert result.plan is not None
    ordered = result.plan.steps
    assert ordered[0].lower().startswith("use")
    assert any("evaluation" in step.lower() for step in ordered)


def test_planning_agent_uses_custom_registry_entries():
    registry = ToolRegistry()
    registry.register(name="VectorRetriever", category="retrieval", description="vector search")
    registry.register(name="TemplateLLMGenerator", category="generation", description="template responses")
    agent = PlanningAgent(registry=registry)

    result = agent.plan("Retrieve knowledge base entries")

    assert result.plan is not None
    assert any("VectorRetriever" in step for step in result.plan.steps)
