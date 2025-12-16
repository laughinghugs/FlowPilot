import pytest

from tools import (
    CompositeRetriever,
    Document,
    HeuristicReranker,
    InMemoryRetriever,
    RagasEvaluator,
)


@pytest.fixture()
def sample_documents():
    return [
        Document(doc_id="1", content="FastAPI enables quick APIs"),
        Document(doc_id="2", content="Retrieval augmented generation uses retrievers"),
        Document(doc_id="3", content="RAG pipelines also use rerankers"),
    ]


def test_in_memory_retriever_returns_scored_documents(sample_documents):
    retriever = InMemoryRetriever(sample_documents)

    results = retriever.retrieve("retrievers", top_k=2)

    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert all(result.doc_id in {"2", "3"} for result in results)


def test_composite_retriever_merges_results(sample_documents):
    retriever = CompositeRetriever([InMemoryRetriever(sample_documents), InMemoryRetriever(sample_documents)])

    results = retriever.retrieve("FastAPI", top_k=1)

    assert len(results) == 1
    assert results[0].doc_id == "1"


def test_reranker_respects_scores(sample_documents):
    retriever = InMemoryRetriever(sample_documents)
    docs = retriever.retrieve("RAG", top_k=3)

    reranker = HeuristicReranker()
    reranked = reranker.rerank("RAG", docs, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].score >= reranked[1].score


def test_ragas_evaluator_supports_custom_loader(sample_documents):
    def fake_loader():
        def fake_evaluate(dataset, metrics):  # noqa: ARG001 - signature parity
            return {"answer_relevancy": 0.5, "context_precision": 0.8}

        class _Dataset:
            @staticmethod
            def from_dict(data):  # noqa: D401
                return data

        return fake_evaluate, ("m1", "m2"), _Dataset

    evaluator = RagasEvaluator(loader=fake_loader)
    scores = evaluator.evaluate("What is RAG?", sample_documents, answer="RAG is a technique")

    assert scores["answer_relevancy"] == 0.5
    assert scores["context_precision"] == 0.8
