"""LLM-backed planning utilities with multi-provider support."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

import requests
from dotenv import load_dotenv

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


@dataclass(frozen=True)
class LLMGeneratedPlan:
    """Structured output from an LLM planner."""

    steps: list[str]
    rationale: str
    clarifying_question: str | None = None


class LLMPlanner(Protocol):
    """Protocol for planner backends."""

    def generate(self, *, user_message: str, registry: ToolRegistry) -> LLMGeneratedPlan:  # pragma: no cover - interface
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
        "Return a JSON object with keys 'plan_steps' (list of step strings), 'rationale', and optionally"
        " 'clarifying_question' if additional input is needed. Keep steps actionable and tool-aware."
    )


def _parse_plan_payload(content: str | None) -> LLMGeneratedPlan:
    data = json.loads(content or "{}")
    steps = [str(step).strip() for step in data.get("plan_steps", []) if str(step).strip()]
    rationale = data.get("rationale", "No rationale provided.")
    clarifying = data.get("clarifying_question")
    return LLMGeneratedPlan(steps=steps, rationale=rationale, clarifying_question=clarifying)


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

    def generate(self, *, user_message: str, registry: ToolRegistry) -> LLMGeneratedPlan:  # noqa: D401
        messages = _build_openai_messages(user_message, registry)
        response = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content
        return _parse_plan_payload(content)


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

    def generate(self, *, user_message: str, registry: ToolRegistry) -> LLMGeneratedPlan:  # noqa: D401
        messages = _build_openai_messages(user_message, registry)
        response = self._client.chat.completions.create(
            model=self._deployment,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return _parse_plan_payload(response.choices[0].message.content)


class AzureFoundryPlanner(LLMPlanner):
    """Planner that calls Azure AI Foundry chat completion endpoint."""

    def __init__(self) -> None:
        self._endpoint = os.getenv("AZURE_FOUNDRY_ENDPOINT")
        self._api_key = os.getenv("AZURE_FOUNDRY_API_KEY")
        self._deployment = os.getenv("AZURE_FOUNDRY_DEPLOYMENT")
        self._api_version = os.getenv("AZURE_FOUNDRY_API_VERSION", "2024-05-01-preview")
        if not all([self._endpoint, self._api_key, self._deployment]):
            raise EnvironmentError("Azure Foundry configuration is incomplete")

    def generate(self, *, user_message: str, registry: ToolRegistry) -> LLMGeneratedPlan:  # noqa: D401
        url = (
            f"{self._endpoint}/openai/deployments/{self._deployment}/chat/completions"
            f"?api-version={self._api_version}"
        )
        payload = {
            "messages": _build_openai_messages(user_message, registry),
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Content-Type": "application/json",
            "api-key": self._api_key,
        }
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_plan_payload(content)


class ClaudePlanner(LLMPlanner):
    """Planner that leverages Anthropic Claude models."""

    def __init__(self, model: str | None = None) -> None:
        if Anthropic is None:  # pragma: no cover
            raise ImportError("anthropic package is required to use ClaudePlanner")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set")

        self._client = Anthropic(api_key=api_key)
        self._model = model or os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20240620")

    def generate(self, *, user_message: str, registry: ToolRegistry) -> LLMGeneratedPlan:  # noqa: D401
        prompt = _build_user_prompt(user_message, registry)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You are an agent-creation strategist. Use available tools to craft a JSON response with"
                " keys 'plan_steps', 'rationale', and optional 'clarifying_question'."
            ),
        )
        content = next((block.text for block in response.content if getattr(block, "text", None)), None)
        return _parse_plan_payload(content)


def _build_openai_messages(user_message: str, registry: ToolRegistry) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an agent-creation strategist. Use GPT-class reasoning to map user goals to available tools."
                " Respond in JSON only."
            ),
        },
        {"role": "user", "content": _build_user_prompt(user_message, registry)},
    ]


def build_planner_from_env() -> LLMPlanner:
    provider = (os.getenv("LLM_PROVIDER") or "openai").lower()

    if provider == "openai":
        return OpenAIPlanner()
    if provider == "azure_openai":
        return AzureOpenAIPlanner()
    if provider == "azure_foundry":
        return AzureFoundryPlanner()
    if provider == "claude":
        return ClaudePlanner()

    raise ValueError(
        "Unsupported LLM_PROVIDER. Expected one of: openai, azure_openai, azure_foundry, claude."
    )
