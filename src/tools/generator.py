"""Text generation utilities for RAG."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from .models import Document


class LLMGenerator(ABC):
    """Interface for wrapping LLM completion APIs."""

    @abstractmethod
    def generate(self, query: str, context: Iterable[Document]) -> str:
        """Produce a response using the query and retrieved context."""


class TemplateLLMGenerator(LLMGenerator):
    """Deterministic generator useful for tests and local development."""

    def __init__(self, template: str | None = None) -> None:
        self._template = template or (
            "Answer the query '{query}' using the following context:\n{context}\n---\nResponse: {answer}"
        )

    def generate(self, query: str, context: Iterable[Document]) -> str:  # noqa: D401
        combined_context = "\n".join(f"- {doc.content}" for doc in context)
        if not combined_context:
            combined_context = "(no supporting documents)"

        return self._template.format(query=query, context=combined_context, answer="TBD")
