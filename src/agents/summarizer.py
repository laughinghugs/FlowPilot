"""Conversation summarization utilities to describe manifest requirements."""

from __future__ import annotations

import os
from typing import Protocol, Sequence

try:  # pragma: no cover - optional dependency
    from openai import AzureOpenAI, OpenAI
except ImportError:  # pragma: no cover
    AzureOpenAI = None  # type: ignore[misc]
    OpenAI = None  # type: ignore[misc]

SUMMARY_SYSTEM_PROMPT = (
    "You are an AI assistant that summarizes planning chats. Produce a concise requirement brief (max 6 sentences) "
    "capturing the user objectives, constraints, and expectations for an agentic pipeline. Use plain language."
)


class ConversationSummarizer(Protocol):
    """Protocol for summarizing chat transcripts."""

    def summarize(self, conversation_history: Sequence[dict[str, str]], *, fallback_text: str) -> str:
        ...


class SimpleSummarizer(ConversationSummarizer):
    """Fallback summarizer when no LLM provider is available."""

    def summarize(self, conversation_history: Sequence[dict[str, str]], *, fallback_text: str) -> str:  # noqa: D401
        for message in reversed(conversation_history or []):
            if message.get("role", "").lower() == "user" and message.get("content"):
                return message["content"].strip()
        return fallback_text


class OpenAISummarizer(ConversationSummarizer):
    """Summarizer powered by OpenAI Chat Completions."""

    def __init__(self, client: OpenAI, model: str | None = None) -> None:
        if OpenAI is None:  # pragma: no cover
            raise ImportError("openai package is required for OpenAISummarizer")
        self._client = client
        self._model = model or os.getenv("OPENAI_SUMMARY_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.1-mini"

    def summarize(self, conversation_history: Sequence[dict[str, str]], *, fallback_text: str) -> str:  # noqa: D401
        if not conversation_history:
            return fallback_text
        messages = _format_history_for_chat(conversation_history)
        chat_messages = [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}] + messages
        response = self._client.chat.completions.create(model=self._model, messages=chat_messages)
        summary = (response.choices[0].message.content or "").strip()
        return summary or fallback_text


class AzureOpenAISummarizer(ConversationSummarizer):
    """Summarizer powered by Azure OpenAI."""

    def __init__(self, client: AzureOpenAI, deployment: str) -> None:
        if AzureOpenAI is None:  # pragma: no cover
            raise ImportError("openai package with Azure support is required for AzureOpenAISummarizer")
        self._client = client
        self._deployment = deployment

    def summarize(self, conversation_history: Sequence[dict[str, str]], *, fallback_text: str) -> str:  # noqa: D401
        if not conversation_history:
            return fallback_text
        messages = _format_history_for_chat(conversation_history)
        chat_messages = [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}] + messages
        response = self._client.chat.completions.create(model=self._deployment, messages=chat_messages)
        summary = (response.choices[0].message.content or "").strip()
        return summary or fallback_text


def build_summarizer_from_env() -> ConversationSummarizer:
    """Instantiate a summarizer based on environment configuration."""
    provider = (os.getenv("LLM_PROVIDER") or "openai").lower()
    try:
        if provider == "openai" and OpenAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                return OpenAISummarizer(OpenAI(api_key=api_key))
        elif provider == "azure_openai" and AzureOpenAI:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            if api_key and endpoint and deployment:
                client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
                return AzureOpenAISummarizer(client, deployment=deployment)
    except Exception:  # pragma: no cover - fallback
        pass
    return SimpleSummarizer()


def _format_history_for_chat(conversation_history: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    formatted: list[dict[str, str]] = []
    for message in conversation_history:
        content = message.get("content", "")
        if not content:
            continue
        role = message.get("role", "").lower()
        if role == "ai":
            mapped_role = "assistant"
        elif role in {"assistant", "system", "user"}:
            mapped_role = role
        else:
            mapped_role = "user"
        formatted.append({"role": mapped_role, "content": content})
    if not formatted:
        formatted.append({"role": "user", "content": ""})
    return formatted


__all__ = [
    "ConversationSummarizer",
    "SimpleSummarizer",
    "OpenAISummarizer",
    "AzureOpenAISummarizer",
    "build_summarizer_from_env",
]
