"""Intent analysis + planning agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

from .llm import CustomToolDefinition, LLMGeneratedPlan, LLMPlanner, PlanStep, build_planner_from_env
from .manifest import PlanManifestEntry, PlanManifestWriter
from .summarizer import ConversationSummarizer, build_summarizer_from_env


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
        planner_backend: LLMPlanner | None = None,
        system_prompt: str | None = None,
        manifest_path: str | None = None,
        summarizer_backend: ConversationSummarizer | None = None,
    ) -> None:
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
            system_prompt=system_prompt or self._system_prompt,
            conversation_history=history_list or None,
        )
        return self._convert_llm_plan(llm_plan)

    def finalize_plan(self, result: PlanningResult, *, fallback_user_message: str) -> str:
        """
        Persist the confirmed plan to the manifest and auto-generate custom tools from the plan steps.

        Args:
            result: PlanningResult from a previous call that includes a finalized plan.
            fallback_user_message: Used when the summarizer cannot infer a requirement brief.
        """
        if not result.plan:
            raise ValueError("Cannot finalize a planning result without a plan.")
        if result.clarifying_question:
            raise ValueError("Cannot finalize a plan while clarifying questions remain.")
        if not self._manifest_writer:
            raise RuntimeError("Manifest writer is not configured; cannot finalize plan.")

        summary = self._summarize_conversation(result.conversation_history, fallback_text=fallback_user_message)
        derived_tools = self._derive_custom_tools_from_plan(result.plan)
        plan_id = self._record_manifest(
            result.plan,
            user_message=summary,
            custom_tools=derived_tools,
        )
        if not plan_id:
            raise RuntimeError("Failed to write manifest entry.")
        return plan_id

    def _convert_llm_plan(self, llm_plan: LLMGeneratedPlan) -> PlanningResult:
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
        return PlanningResult(
            plan=plan,
            clarifying_question=llm_plan.clarifying_question,
            conversation_history=history,
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

    def _derive_custom_tools_from_plan(self, plan: AgentPlan) -> tuple[CustomToolDefinition, ...]:
        """Create deterministic custom tool definitions from the plan steps."""
        derived: list[CustomToolDefinition] = []
        seen: set[str] = set()
        for step in plan.steps:
            if step.tool in seen:
                continue
            seen.add(step.tool)
            metadata = dict(step.metadata)
            inputs: str | dict[str, list[str]]
            if metadata:
                inputs = {"expected_fields": sorted(metadata.keys())}
            else:
                inputs = "Accept the user's query and any shared context dict."
            derived.append(
                CustomToolDefinition(
                    name=step.tool,
                    purpose=step.rationale or f"Auto-generated tool for {step.tool}",
                    inputs=inputs,
                    data_sources=None,
                    credentials=None,
                    metadata={"derived_from_plan": True, "plan_metadata": metadata},
                )
            )
        return tuple(derived)


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
