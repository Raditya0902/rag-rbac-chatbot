"""Unit tests for hybrid retrieval and cross-encoder re-ranking.

These tests mock the cross-encoder to avoid downloading the model at test time.
They exercise rerank_docs() logic only — full integration tests (hitting ChromaDB)
are left for manual / CI-with-vectorstore runs.
"""
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.vectorstore import store


def _doc(content: str, namespace: str = "general") -> Document:
    return Document(page_content=content, metadata={"namespace": namespace, "source": "test.md"})


def _mock_encoder(scores: list[float], monkeypatch) -> MagicMock:
    """Patch _get_cross_encoder so rerank_docs uses our mock."""
    mock = MagicMock()
    mock.predict.return_value = scores
    monkeypatch.setattr(store, "_get_cross_encoder", lambda: mock)
    return mock


# ---------------------------------------------------------------------------
# rerank_docs — basic behaviour
# ---------------------------------------------------------------------------

def test_rerank_returns_top_k(monkeypatch):
    docs = [_doc(f"doc {i}") for i in range(5)]
    _mock_encoder([5.0, 4.0, 3.0, 2.0, 1.0], monkeypatch)  # doc 0 gets highest score

    result = store.rerank_docs("test query", docs, top_k=3)

    assert len(result) == 3
    assert result[0].page_content == "doc 0"


def test_rerank_empty_input_returns_empty(monkeypatch):
    result = store.rerank_docs("query", [], top_k=5)
    assert result == []


def test_rerank_reverses_order_when_scores_are_reversed(monkeypatch):
    docs = [_doc("A"), _doc("B"), _doc("C")]
    # C (index 2) gets highest score — should rank first
    _mock_encoder([1.0, 2.0, 3.0], monkeypatch)

    result = store.rerank_docs("query", docs, top_k=3)

    assert result[0].page_content == "C"
    assert result[1].page_content == "B"
    assert result[2].page_content == "A"


def test_rerank_top_k_larger_than_docs_returns_all(monkeypatch):
    docs = [_doc("X"), _doc("Y")]
    _mock_encoder([2.0, 1.0], monkeypatch)

    result = store.rerank_docs("query", docs, top_k=10)

    assert len(result) == 2


def test_rerank_top_k_one_returns_single_best(monkeypatch):
    docs = [_doc("low"), _doc("high"), _doc("mid")]
    _mock_encoder([1.0, 9.0, 5.0], monkeypatch)

    result = store.rerank_docs("query", docs, top_k=1)

    assert len(result) == 1
    assert result[0].page_content == "high"


def test_rerank_calls_encoder_with_correct_pairs(monkeypatch):
    docs = [_doc("alpha"), _doc("beta")]
    mock = _mock_encoder([1.0, 2.0], monkeypatch)

    store.rerank_docs("my query", docs, top_k=2)

    mock.predict.assert_called_once_with([("my query", "alpha"), ("my query", "beta")])


# ---------------------------------------------------------------------------
# rerank_docs — metadata preservation
# ---------------------------------------------------------------------------

def test_rerank_preserves_metadata(monkeypatch):
    docs = [
        _doc("finance doc", namespace="finance"),
        _doc("hr doc", namespace="hr"),
    ]
    _mock_encoder([2.0, 1.0], monkeypatch)

    result = store.rerank_docs("query", docs, top_k=2)

    assert result[0].metadata["namespace"] == "finance"
    assert result[1].metadata["namespace"] == "hr"
