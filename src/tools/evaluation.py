"""RAG evaluation helpers (integrates with ragas when available)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Sequence

from .models import Document

LoaderType = Callable[[], tuple[Callable[..., dict[str, float]], Sequence[object], type]]


class RagEvaluator(ABC):
    """Common interface for evaluating RAG responses."""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        documents: Iterable[Document],
        answer: str,
        ground_truth: str | None = None,
    ) -> dict[str, float]:
        """Return metric scores for the supplied RAG artefacts."""


class RagasUnavailableError(RuntimeError):
    """Raised when ragas or its dependencies cannot be imported."""


class RagasEvaluator(RagEvaluator):
    """Thin wrapper around the ragas evaluation pipeline."""

    def __init__(self, loader: LoaderType | None = None) -> None:
        self._loader = loader or self._default_loader

    def evaluate(
        self,
        query: str,
        documents: Iterable[Document],
        answer: str,
        ground_truth: str | None = None,
    ) -> dict[str, float]:  # noqa: D401
        ragas_evaluate, metrics, dataset_cls = self._loader()

        dataset = dataset_cls.from_dict(
            {
                "question": [query],
                "contexts": [[doc.content for doc in documents]],
                "answer": [answer],
                "ground_truth": [ground_truth or ""],
            }
        )

        result = ragas_evaluate(dataset, metrics=list(metrics))
        return {name: float(score) for name, score in result.items()}

    @staticmethod
    def _default_loader() -> tuple[Callable[..., dict[str, float]], Sequence[object], type]:
        try:
            from datasets import Dataset
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import answer_relevancy, context_precision, faithfulness
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise RagasUnavailableError(
                "ragas and datasets packages are required for RagasEvaluator"
            ) from exc

        metrics = [answer_relevancy, context_precision, faithfulness]
        return ragas_evaluate, metrics, Dataset
