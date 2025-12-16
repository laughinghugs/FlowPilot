"""Shared data structures for RAG tooling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Lightweight representation of a chunk of text."""

    doc_id: str
    content: str
    score: float | None = None
