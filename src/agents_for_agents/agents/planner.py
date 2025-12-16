"""Intent analysis + planning agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from agents_for_agents.tools import (
    CompositeRetriever,
    HeuristicReranker,
    InMemoryRetriever,
    LLMGenerator,
    RagEvaluator,
    RagasEvaluator,
    Reranker,
    TemplateLLMGenerator,
)


@dataclass(frozen=True)
class ToolCapability:
    """Describe what a tool can do."""

    name: str
    category: str
    description: str


@dataclass(frozen=True)
class ToolInventory:
    """Collection of available tools grouped by capability."""

    tools: Sequence[ToolCapability]

    def categories(self) -> set[str]:
        return {tool.category for tool in self.tools}

    def get_tool(self, category: str) -> ToolCapability | None:
        for tool in self.tools:
            if tool.category == category:
                return tool
        return None

    @classmethod
    def default(cls) -> "ToolInventory":
        return cls.from_registry(DEFAULT_TOOL_REGISTRY)

    @classmethod
    def from_registry(cls, registry: "ToolRegistry") -> "ToolInventory":
        return cls(tools=registry.capabilities())


class ToolRegistry:
    """Registry that stores tool capabilities and allows user extensions."""

    def __init__(self, capabilities: Iterable[ToolCapability] | None = None) -> None:
        self._capabilities: list[ToolCapability] = list(capabilities or [])

    def register(self, *, name: str, category: str, description: str) -> ToolCapability:
        capability = ToolCapability(name=name, category=category, description=description)
        self._capabilities.append(capability)
        return capability

    def capabilities(self) -> Sequence[ToolCapability]:
        return tuple(self._capabilities)

    def categories(self) -> set[str]:
        return {capability.category for capability in self._capabilities}

    @classmethod
    def with_default_tools(cls) -> "ToolRegistry":
        registry = cls()
        registry.register(
            name=InMemoryRetriever.__name__,
            category="retrieval",
            description="keyword-based recall over in-memory documents",
        )
        registry.register(
            name=CompositeRetriever.__name__,
            category="retrieval",
            description="combine recall strategies for better coverage",
        )
        registry.register(
            name=TemplateLLMGenerator.__name__,
            category="generation",
            description="deterministic template LLM responses",
        )
        registry.register(
            name=LLMGenerator.__name__,
            category="generation",
            description="LLM API wrapper (abstract)",
        )
        registry.register(
            name=HeuristicReranker.__name__,
            category="rerank",
            description="heuristic-driven reranking",
        )
        registry.register(
            name=Reranker.__name__,
            category="rerank",
            description="reranking interface",
        )
        registry.register(
            name=RagasEvaluator.__name__,
            category="evaluation",
            description="ragas-based evaluation",
        )
        registry.register(
            name=RagEvaluator.__name__,
            category="evaluation",
            description="evaluation interface",
        )
        return registry


DEFAULT_TOOL_REGISTRY = ToolRegistry.with_default_tools()


@dataclass(frozen=True)
class AgentPlan:
    """Structured plan for assembling an AI agent."""

    steps: Sequence[str]
    rationale: str


@dataclass(frozen=True)
class PlanningResult:
    """Outcome of running the planning agent."""

    plan: AgentPlan | None = None
    clarifying_question: str | None = None

    def require_plan(self) -> AgentPlan:
        if self.plan is None:
            raise ValueError("PlanningResult does not contain a plan")
        return self.plan


class PlanningAgent:
    """Analyzes user intent and maps it to available toolchains."""

    def __init__(
        self,
        inventory: ToolInventory | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        if inventory and registry:
            raise ValueError("Provide either inventory or registry, not both.")

        if inventory is not None:
            self._inventory = inventory
        else:
            selected_registry = registry or DEFAULT_TOOL_REGISTRY
            self._inventory = ToolInventory.from_registry(selected_registry)

    def plan(self, user_message: str) -> PlanningResult:
        intent = self._analyze_intent(user_message)
        missing = intent.required_categories - self._inventory.categories()

        if missing:
            question = self._build_clarifying_question(missing)
            return PlanningResult(clarifying_question=question)

        steps = self._build_steps(intent)
        rationale = self._summarize_rationale(intent)
        return PlanningResult(plan=AgentPlan(steps=steps, rationale=rationale))

    def _analyze_intent(self, message: str) -> "Intent":
        lowered = message.lower()
        required: set[str] = set()

        keywords = {
            "retrieval": ["retrieve", "search", "document", "knowledge", "rag"],
            "generation": ["generate", "respond", "answer", "llm"],
            "rerank": ["rerank", "re-rank", "rank"],
            "evaluation": ["evaluate", "metrics", "ragas", "quality"],
        }

        for category, triggers in keywords.items():
            if any(token in lowered for token in triggers):
                required.add(category)

        # default pipeline if not explicitly stated
        if not required:
            required.update({"retrieval", "generation", "rerank"})

        return Intent(raw_message=message, required_categories=required)

    def _build_steps(self, intent: "Intent") -> List[str]:
        steps: list[str] = []
        for category in intent.ordered_categories():
            tool = self._inventory.get_tool(category)
            description = tool.description if tool else "matching capability"
            steps.append(f"Use {tool.name if tool else category} for {description}.")
        steps.append("Integrate steps into an agentic workflow and expose via your preferred interface.")
        return steps

    def _summarize_rationale(self, intent: "Intent") -> str:
        categories = ", ".join(sorted(intent.required_categories))
        return f"Plan covers the following capabilities requested or inferred: {categories}."

    @staticmethod
    def _build_clarifying_question(missing: set[str]) -> str:
        categories = ", ".join(sorted(missing))
        return (
            "I do not currently have tooling for the following capabilities: "
            f"{categories}. Could you provide more details or alternative requirements?"
        )


@dataclass(frozen=True)
class Intent:
    raw_message: str
    required_categories: set[str] = field(default_factory=set)

    def ordered_categories(self) -> Sequence[str]:
        order = ["retrieval", "rerank", "generation", "evaluation"]
        ordered = [category for category in order if category in self.required_categories]
        others = [category for category in self.required_categories if category not in order]
        return [*ordered, *sorted(others)]
