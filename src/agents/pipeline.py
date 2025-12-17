"""Pipeline execution agent that replays manifest plans using manifest-defined tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence

from .llm import CustomToolDefinition, PlanStep
from .manifest import PlanManifestReader


class ToolNotFoundError(RuntimeError):
    """Raised when a plan references a tool without an executor."""


class DefaultToolResolver:
    """
    Resolves tools purely from manifest descriptions.

    Built-in tool implementations have been removed; every step is treated as a manifest-defined tool. If the plan
    exposes an entry under `custom_tools`, that definition is used; otherwise a generic payload is emitted so
    downstream systems can inspect or implement the tool later.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[PlanStep, dict[str, Any]], Any]] = {}
        self._custom_tools: Dict[str, CustomToolDefinition] = {}

    def register_custom_tools(self, tools: Sequence[CustomToolDefinition]) -> None:
        """Refresh the resolver's view of available custom tool definitions."""
        self._custom_tools = {tool.name: tool for tool in tools}

    def execute(self, step: PlanStep, context: dict[str, Any]) -> Any:
        handler = self._handlers.get(step.tool)
        if handler is not None:
            return handler(step, context)

        custom_tool = self._custom_tools.get(step.tool)
        if custom_tool:
            return self._run_custom_tool(step, context, custom_tool)

        return self._run_dynamic_tool(step, context)

    @staticmethod
    def _run_custom_tool(step: PlanStep, context: dict[str, Any], definition: CustomToolDefinition) -> dict[str, Any]:
        payload = {
            "tool": definition.name,
            "purpose": definition.purpose,
            "inputs": definition.inputs,
            "data_sources": definition.data_sources,
            "credentials": definition.credentials,
            "definition_metadata": definition.metadata,
            "execution_metadata": step.metadata,
        }
        output_key = step.metadata.get("output") or f"{definition.name}_result"
        context[output_key] = payload
        return payload

    @staticmethod
    def _run_dynamic_tool(step: PlanStep, context: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "tool": step.tool,
            "rationale": step.rationale,
            "metadata": step.metadata,
            "status": "manifest_defined",
        }
        output_key = step.metadata.get("output") or f"{step.tool}_result"
        context[output_key] = payload
        return payload


@dataclass
class PipelineResult:
    context: dict[str, Any]
    plan_id: str


class PipelineAgent:
    """Executes manifest-defined plans using only the tool definitions captured in the manifest."""

    def __init__(
        self,
        manifest_path: str = "plan_manifests.jsonl",
        resolver: DefaultToolResolver | None = None,
    ) -> None:
        self._reader = PlanManifestReader(manifest_path)
        self._resolver = resolver or DefaultToolResolver()

    def execute(self, plan_id: str, context: dict[str, Any] | None = None) -> PipelineResult:
        entry = self._reader.get(plan_id)
        ctx = dict(context or {})

        if hasattr(self._resolver, "register_custom_tools"):
            self._resolver.register_custom_tools(entry.custom_tools)

        for step in entry.steps:
            self._resolver.execute(step, ctx)

        return PipelineResult(context=ctx, plan_id=plan_id)
