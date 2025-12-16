from agents import PlanningAgent, ToolRegistry
from agents.llm import LLMGeneratedPlan


class StubPlanner:
    def __init__(self) -> None:
        self.plan = LLMGeneratedPlan(
            steps=[
                "Use InMemoryRetriever for recall",
                "Apply HeuristicReranker",
                "Generate response via TemplateLLMGenerator",
            ],
            rationale="Covers retrieval, reranking, generation",
        )

    def generate(  # noqa: D401
        self,
        *,
        user_message: str,
        registry: ToolRegistry,
        system_prompt: str | None = None,
    ) -> LLMGeneratedPlan:
        assert system_prompt is None
        return self.plan


def test_planning_agent_generates_plan_for_rag_request():
    agent = PlanningAgent(planner_backend=StubPlanner())

    result = agent.plan("Need retrieval, reranking, and generation for a RAG agent.")

    assert result.plan is not None
    assert result.clarifying_question is None
    assert len(result.plan.steps) == 3
