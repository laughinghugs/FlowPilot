from agents_for_agents.agents import PlanningAgent


def test_planning_agent_generates_plan_for_rag_request():
    agent = PlanningAgent()

    result = agent.plan("Need retrieval, reranking, and generation for a RAG agent.")

    assert result.plan is not None
    assert result.clarifying_question is None
    assert len(result.plan.steps) >= 3
