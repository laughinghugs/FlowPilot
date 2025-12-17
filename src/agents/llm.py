"""LLM-backed planning utilities with multi-provider support."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

try:  # pragma: no cover - optional at test time
    from openai import AzureOpenAI, OpenAI
except ImportError:  # pragma: no cover
    AzureOpenAI = None  # type: ignore[misc]
    OpenAI = None  # type: ignore[misc]

try:  # pragma: no cover - optional
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore[misc]

from .registry import ToolRegistry

load_dotenv()

DEFAULT_SYSTEM_PROMPT = (
    "You are an agent-creation strategist. Review the registered tools first. "
    "If they are sufficient, outline a plan using only those tools. "
    "When no existing tool can satisfy a required capability, describe a custom tool concept, "
    "collecting the input format, data sources to connect, and any credentials or API keys needed. "
    "Ask follow-up questions in clear, non-technical language to gather those details before finalizing the plan. "
    "Always respond in JSON."
)


def _resolve_system_prompt(system_prompt: str | None) -> str:
    return system_prompt.strip() if system_prompt else DEFAULT_SYSTEM_PROMPT


class PlanStepModel(BaseModel):
    tool: str
    rationale: str
    metadata: dict[str, Any] = Field(default_factory=dict)


JSONLike = dict[str, Any]


class CustomToolModel(BaseModel):
    name: str
    purpose: str
    inputs: str | JSONLike
    data_sources: str | JSONLike | None = None
    credentials: str | JSONLike | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlannerResponseModel(BaseModel):
    plan: list[PlanStepModel] = Field(default_factory=list)
    clarifying_questions: str | None = None
    custom_tools: list[CustomToolModel] = Field(default_factory=list)


@dataclass(frozen=True)
class PlanStep:
    tool: str
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMGeneratedPlan:
    """Structured output from an LLM planner."""

    steps: list[PlanStep]
    clarifying_question: str | None = None
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    custom_tools: list["CustomToolDefinition"] = field(default_factory=list)


@dataclass(frozen=True)
class CustomToolDefinition:
    name: str
    purpose: str
    inputs: str | JSONLike
    data_sources: str | JSONLike | None = None
    credentials: str | JSONLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_history_for_model(
    conversation_history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """
    Convert external conversation roles to the roles expected by provider SDKs.
    External format uses 'AI' for assistant responses; provider SDKs expect 'assistant'.
    This returns a new list with role normalized: 'AI' -> 'assistant'. Keeps 'system' and 'user'.
    """
    if not conversation_history:
        return []
    normalized: list[dict[str, str]] = []
    for msg in conversation_history:
        role = msg.get("role", "").lower()
        if role == "ai":
            provider_role = "assistant"
        elif role in ("assistant", "system", "user"):
            provider_role = role
        else:
            # unknown role - preserve as-is (safer) but lowercased
            provider_role = role
        normalized.append({"role": provider_role, "content": msg.get("content", "")})
    return normalized

def _build_openai_messages(
    user_message: str | None,
    registry: ToolRegistry,
    system_prompt: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """
    Build messages list with a single system prompt (resolved from param, history, or default),
    then the conversation history (normalized), and finally the current user message (if any).

    Accepts external conversation format where assistant messages may have role "AI".
    """
    messages: list[dict[str, str]] = []

    # If system_prompt provided explicitly, use it; otherwise try to find first system message
    resolved_system: str | None = None
    if system_prompt:
        resolved_system = _resolve_system_prompt(system_prompt)
    elif conversation_history:
        # find first system message in conversation_history
        for msg in conversation_history:
            if msg.get("role", "").lower() == "system":
                resolved_system = msg.get("content", "").strip()
                break

    # Ensure there's always a system message
    messages.append({"role": "system", "content": resolved_system or _resolve_system_prompt(None)})

    # Normalize conversation history and append non-system messages (we've already handled system)
    normalized = _normalize_history_for_model(conversation_history)
    for msg in normalized:
        if msg["role"] == "system":
            # skip duplicate system entries (we already placed the system message at start)
            continue
        messages.append(msg)

    # Add current user message (as full prompt with tool list) if provided
    if user_message:
        messages.append({"role": "user", "content": _build_user_prompt(user_message, registry)})

    return messages


def _build_external_history(
    messages: list[dict[str, str]],
    assistant_content: str,
) -> list[dict[str, str]]:
    """Map provider roles back to external history format with 'AI' assistant entries."""
    if not messages:
        return [{"role": "AI", "content": assistant_content}]

    external: list[dict[str, str]] = []
    first = messages[0]
    if first.get("role") == "system":
        external.append({"role": "system", "content": first.get("content", "")})
    else:
        external.append({"role": first.get("role", "system"), "content": first.get("content", "")})

    for msg in messages[1:]:
        role = msg.get("role", "")
        if role == "assistant":
            external_role = "AI"
        else:
            external_role = role
        external.append({"role": external_role, "content": msg.get("content", "")})

    external.append({"role": "AI", "content": assistant_content})
    return external


class LLMPlanner(Protocol):
    """Protocol for planner backends."""

    def generate(  # pragma: no cover - interface
        self,
        *,
        user_message: str | None = None,
        registry: ToolRegistry,
        system_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> LLMGeneratedPlan:
        ...


def _build_user_prompt(user_message: str, registry: ToolRegistry) -> str:
    tool_lines = [
        f"- {capability.name} [{capability.category}]: {capability.description}"
        for capability in registry.capabilities()
    ]
    tool_summary = "\n".join(tool_lines) or "(no tools registered)"

    return (
        "User request:\n"
        f"{user_message}\n\n"
        "Available tools (choose only from these capabilities):\n"
        f"{tool_summary}\n\n"
        "Return JSON: {\n"
        '  "plan": [\n'
        '    {"tool": "<tool name>", "rationale": "<why this tool>", "metadata": {"param": "value"}}\n'
        "  ],\n"
        '  "clarifying_questions": "<string if more info needed, else null>",\n'
        '  "custom_tools": [\n'
        '    {\n'
        '      "name": "<proposed tool>",\n'
        '      "purpose": "<plain-language summary>",\n'
        '      "inputs": "<what information the tool expects (string or JSON)>",\n'
        '      "data_sources": "<APIs, databases, or files to connect (string or JSON)>",\n'
        '      "credentials": "<api keys or auth needed (string or JSON)>,"\n'
        '      "metadata": {"linked_plan_step": "<tool usage context>"}\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Only add entries to 'custom_tools' when no existing tool suffices. Ask clarifying questions in simple language"
        " to gather any required inputs, data sources, or credentials. Each 'tool' in the plan must be either an existing"
        " capability or reference one of the newly defined custom tools explicitly."
    )


def _parse_plan_payload(content: str | None, registry: ToolRegistry) -> LLMGeneratedPlan:
    try:
        response = PlannerResponseModel.model_validate_json(content or "{}")
    except ValidationError as exc:  # pragma: no cover - provider specific
        raise ValueError(f"Planner response is invalid: {exc}") from exc

    tool_names = {capability.name for capability in registry.capabilities()}
    steps: list[PlanStep] = []
    for step in response.plan:
        if step.tool in tool_names:
            # raise ValueError(f"Planner referenced unknown tool '{step.tool}'")
            steps.append(PlanStep(tool=step.tool, rationale=step.rationale, metadata=dict(step.metadata)))

    custom_tools = [
        CustomToolDefinition(
            name=tool.name,
            purpose=tool.purpose,
            inputs=tool.inputs,
            data_sources=tool.data_sources,
            credentials=tool.credentials,
            metadata=dict(tool.metadata),
        )
        for tool in response.custom_tools
    ]

    clarifying = response.clarifying_questions
    return LLMGeneratedPlan(steps=steps, clarifying_question=clarifying, custom_tools=custom_tools)


class OpenAIPlanner(LLMPlanner):
    """Concrete planner that leverages OpenAI GPT-5 models."""

    def __init__(self, model: str | None = None) -> None:
        if OpenAI is None:  # pragma: no cover
            raise ImportError("openai package is required to use OpenAIPlanner")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your environment or a .env file."
            )

        self._client = OpenAI(api_key=api_key)
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-5.1-mini")

    def generate(  # noqa: D401
        self,
        *,
        user_message: str | None = None,
        registry: ToolRegistry,
        system_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> LLMGeneratedPlan:
        if not user_message and not conversation_history:
            raise ValueError("Either user_message or conversation_history must be provided")

        messages = _build_openai_messages(user_message, registry, system_prompt, conversation_history)
        response = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content
        parsed = _parse_plan_payload(content, registry)
        external_history = _build_external_history(messages, content)

        return LLMGeneratedPlan(
            steps=parsed.steps,
            clarifying_question=parsed.clarifying_question,
            conversation_history=external_history,
        )


class AzureOpenAIPlanner(LLMPlanner):
    """Planner backed by Azure OpenAI (standard Azure OpenAI resource)."""

    def __init__(self, deployment: str | None = None) -> None:
        if AzureOpenAI is None:  # pragma: no cover
            raise ImportError("openai package with Azure support is required")

        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
        deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not all([api_key, endpoint, deployment]):
            raise EnvironmentError("Azure OpenAI configuration is incomplete")

        self._client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
        self._deployment = deployment

    def generate(  # noqa: D401
        self,
        *,
        user_message: str | None = None,
        registry: ToolRegistry,
        system_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> LLMGeneratedPlan:
        if not user_message and not conversation_history:
            raise ValueError("Either user_message or conversation_history must be provided")

        messages = _build_openai_messages(user_message, registry, system_prompt, conversation_history)
        response = self._client.chat.completions.create(
            model=self._deployment,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content
        parsed = _parse_plan_payload(content, registry)
        external_history = _build_external_history(messages, content)

        return LLMGeneratedPlan(
            steps=parsed.steps,
            clarifying_question=parsed.clarifying_question,
            conversation_history=external_history,
        )

def build_planner_from_env() -> LLMPlanner:
    provider = (os.getenv("LLM_PROVIDER") or "openai").lower()

    if provider == "openai":
        return OpenAIPlanner()
    if provider == "azure_openai":
        return AzureOpenAIPlanner()

    raise ValueError(
        "Unsupported LLM_PROVIDER. Expected one of: openai, azure_openai, azure_foundry, claude."
    )
