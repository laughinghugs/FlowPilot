"""Intent analysis + planning agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .llm import LLMGeneratedPlan, LLMPlanner, PlanStep, build_planner_from_env
from .registry import DEFAULT_TOOL_REGISTRY, ToolCapability, ToolRegistry


@dataclass(frozen=True)
class ToolInventory:
    """Collection of available tools grouped by capability."""

    tools: Sequence[ToolCapability]

    def categories(self) -> set[str]:
        return {tool.category for tool in self.tools}

    def get_tool(self, category: str):
        for tool in self.tools:
            if tool.category == category:
                return tool
        return None

    @classmethod
    def default(cls) -> "ToolInventory":
        return cls.from_registry(DEFAULT_TOOL_REGISTRY)

    @classmethod
    def from_registry(cls, registry: ToolRegistry) -> "ToolInventory":
        return cls(tools=registry.capabilities())


@dataclass(frozen=True)
class AgentPlan:
    """Structured plan for assembling an AI agent."""

    steps: Sequence[PlanStep]


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
    """Delegates plan generation to an LLM backend (GPT-5 by default)."""

    def __init__(
        self,
        inventory: ToolInventory | None = None,
        registry: ToolRegistry | None = None,
        planner_backend: LLMPlanner | None = None,
        system_prompt: str | None = None,
    ) -> None:
        if inventory and registry:
            raise ValueError("Provide either inventory or registry, not both.")

        if inventory is not None:
            self._inventory = inventory
            self._registry = registry or ToolRegistry(capabilities=inventory.tools)
        else:
            self._registry = registry or DEFAULT_TOOL_REGISTRY
            self._inventory = ToolInventory.from_registry(self._registry)
        self._planner = planner_backend or build_planner_from_env()
        self._system_prompt = system_prompt

    def plan(self, user_message: str) -> PlanningResult:
        llm_plan = self._planner.generate(
            user_message=user_message,
            registry=self._registry,
            system_prompt=self._system_prompt,
        )
        return self._convert_llm_plan(llm_plan)

    def _convert_llm_plan(self, llm_plan: LLMGeneratedPlan) -> PlanningResult:
        if not llm_plan.steps and llm_plan.clarifying_question:
            return PlanningResult(clarifying_question=llm_plan.clarifying_question)

        if not llm_plan.steps:
            raise ValueError("LLM did not return plan steps or clarification")

        plan = AgentPlan(steps=tuple(llm_plan.steps))
        return PlanningResult(plan=plan, clarifying_question=llm_plan.clarifying_question)
