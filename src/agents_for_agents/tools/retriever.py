"""Retriever implementations for RAG pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from .models import Document


class Retriever(ABC):
    """Abstract retriever contract."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Return the most relevant documents for a query."""


class InMemoryRetriever(Retriever):
    """Simple retriever that ranks in-memory documents via keyword overlap."""

    def __init__(self, documents: Iterable[Document]) -> None:
        self._documents: list[Document] = list(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:  # noqa: D401
        normalized_query = query.lower()
        ranked: list[Document] = []

        for doc in self._documents:
            score = self._score_document(doc.content.lower(), normalized_query)
            ranked.append(Document(doc_id=doc.doc_id, content=doc.content, score=score))

        ranked.sort(key=lambda document: document.score or 0.0, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _score_document(document_text: str, query: str) -> float:
        if not query.strip():
            return 0.0
        return sum(1 for token in query.split() if token in document_text)


class CompositeRetriever(Retriever):
    """Chains multiple retrievers to widen recall."""

    def __init__(self, retrievers: Sequence[Retriever]) -> None:
        if not retrievers:
            raise ValueError("At least one retriever must be provided")
        self._retrievers = list(retrievers)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:  # noqa: D401
        seen: dict[str, Document] = {}
        for retriever in self._retrievers:
            for document in retriever.retrieve(query, top_k=top_k):
                existing = seen.get(document.doc_id)
                if existing is None or (document.score or 0) > (existing.score or 0):
                    seen[document.doc_id] = document

        ranked = sorted(seen.values(), key=lambda doc: doc.score or 0.0, reverse=True)
        return ranked[:top_k]
