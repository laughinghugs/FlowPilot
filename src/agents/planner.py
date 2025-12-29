"""Intent analysis + planning agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

from .llm import CustomToolDefinition, LLMGeneratedPlan, LLMPlanner, PlanStep, build_planner_from_env
from .manifest import PlanManifestEntry, PlanManifestWriter
from .registry import DEFAULT_TOOL_REGISTRY, ToolCapability, ToolRegistry
from .summarizer import ConversationSummarizer, build_summarizer_from_env


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
    conversation_history: tuple[dict[str, str], ...] = tuple()
    plan_id: str | None = None
    custom_tools: tuple[CustomToolDefinition, ...] = tuple()

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
        manifest_path: str | None = None,
        summarizer_backend: ConversationSummarizer | None = None,
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
        self._summarizer = summarizer_backend or build_summarizer_from_env()
        resolved_manifest_path = manifest_path or os.getenv("PLAN_MANIFEST_PATH", "plan_manifests.jsonl")
        self._manifest_writer = PlanManifestWriter(resolved_manifest_path) if resolved_manifest_path else None

    def plan(
        self,
        user_message: str | None = None,
        *,
        conversation_history: Sequence[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> PlanningResult:
        """
        Generate or refine a plan based on the latest user message and optional chat history.

        Args:
            user_message: Most recent user instruction. Optional if provided in conversation_history.
            conversation_history: Existing history of role/content dicts (system/user/AI).
            system_prompt: Optional override for the system prompt on this call.
        """
        history_list = list(conversation_history or [])
        effective_user_message = (user_message or _latest_user_message(history_list))
        if not effective_user_message:
            raise ValueError(
                "PlanningAgent.plan requires a user_message or a conversation_history containing a user entry."
            )

        llm_plan = self._planner.generate(
            user_message=effective_user_message,
            registry=self._registry,
            system_prompt=system_prompt or self._system_prompt,
            conversation_history=history_list or None,
        )
        return self._convert_llm_plan(llm_plan, user_message=effective_user_message)

    def _convert_llm_plan(self, llm_plan: LLMGeneratedPlan, *, user_message: str) -> PlanningResult:
        history = tuple(llm_plan.conversation_history)
        custom_tools = tuple(llm_plan.custom_tools)
        if not llm_plan.steps and llm_plan.clarifying_question:
            return PlanningResult(
                clarifying_question=llm_plan.clarifying_question,
                conversation_history=history,
                custom_tools=custom_tools,
            )

        if not llm_plan.steps:
            raise ValueError("LLM did not return plan steps or clarification")

        plan = AgentPlan(steps=tuple(llm_plan.steps))
        plan_id = None
        if llm_plan.clarifying_question is None:
            summary = self._summarize_conversation(history, fallback_text=user_message)
            plan_id = self._record_manifest(
                plan,
                user_message=summary,
                custom_tools=llm_plan.custom_tools,
            )
        return PlanningResult(
            plan=plan,
            clarifying_question=llm_plan.clarifying_question,
            conversation_history=history,
            plan_id=plan_id,
            custom_tools=custom_tools,
        )

    def _record_manifest(
        self,
        plan: AgentPlan,
        *,
        user_message: str,
        custom_tools: Sequence[CustomToolDefinition] | None = None,
    ) -> str | None:
        if not self._manifest_writer:
            return None
        entry = PlanManifestEntry.create(
            user_message=user_message,
            steps=plan.steps,
            system_prompt=self._system_prompt,
            custom_tools=custom_tools,
        )
        self._manifest_writer.write(entry)
        return entry.plan_id

    def _summarize_conversation(
        self,
        conversation_history: Sequence[dict[str, str]],
        *,
        fallback_text: str,
    ) -> str:
        try:
            return self._summarizer.summarize(conversation_history, fallback_text=fallback_text)
        except Exception:
            return fallback_text


def _latest_user_message(history: Sequence[dict[str, str]] | None) -> str | None:
    """Return the most recent non-empty user message from the conversation history."""
    if not history:
        return None
    for message in reversed(history):
        if message.get("role", "").lower() == "user":
            content = message.get("content", "").strip()
            if content:
                return content
    return None
