"""Reranking helpers for RAG pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from .models import Document


class Reranker(ABC):
    """Interface for reranking retrieved documents."""

    @abstractmethod
    def rerank(self, query: str, documents: Iterable[Document], top_k: int | None = None) -> list[Document]:
        """Return reranked documents ordered by relevance."""


class HeuristicReranker(Reranker):
    """Reranks documents using simple heuristic scoring.

    This implementation boosts documents that already include the entire query
    string and then uses length as a proxy for information density. It is meant
    as a deterministic, dependency-free baseline for testing.
    """

    def rerank(
        self,
        query: str,
        documents: Iterable[Document],
        top_k: int | None = None,
    ) -> list[Document]:  # noqa: D401
        normalized_query = query.lower().strip()
        scored: list[Document] = []
        for doc in documents:
            score = (doc.score or 0.0) + self._bonus(normalized_query, doc.content.lower())
            scored.append(Document(doc_id=doc.doc_id, content=doc.content, score=score))

        scored.sort(key=lambda document: document.score or 0.0, reverse=True)
        if top_k is None:
            return scored
        return scored[:top_k]

    @staticmethod
    def _bonus(query: str, content: str) -> float:
        if not query:
            return 0.0
        if query in content:
            return 1.0
        return 0.0
