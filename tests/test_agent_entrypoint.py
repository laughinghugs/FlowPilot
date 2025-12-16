from agents import PlanningAgent, ToolRegistry
from agents.llm import LLMGeneratedPlan, PlanStep


class StubPlanner:
    def __init__(self) -> None:
        self.plan = LLMGeneratedPlan(
            steps=[
                PlanStep(tool="InMemoryRetriever", rationale="Recall", metadata={"top_k": 3}),
                PlanStep(tool="HeuristicReranker", rationale="Sort results", metadata={}),
                PlanStep(tool="TemplateLLMGenerator", rationale="Respond", metadata={}),
            ]
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
