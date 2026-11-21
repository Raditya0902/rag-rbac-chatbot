"""Cross-department isolation: a role never retrieves chunks it can't access.

Retrieval is a BM25 + vector ensemble, so both halves are checked. The BM25
tests build each role's index from the real documents in resources/data. The
vector-store tests run against the ingested ChromaDB and are skipped when it
hasn't been built (python -m src.ingestion.ingest).
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.auth.rbac import ROLE_NAMESPACE_MAP
from src.config import settings
from src.vectorstore import store

# Queries worded to pull in departments the role should not see.
CROSS_DEPARTMENT_QUERIES = [
    "What was the quarterly revenue, gross margin and cash flow?",           # finance
    "What is the salary, payroll, bonus and maternity leave policy?",         # hr
    "What was the marketing campaign budget, ROI and customer acquisition?",  # marketing
    "What is the system architecture, tech stack and deployment pipeline?",   # engineering
]

ROLES = sorted(ROLE_NAMESPACE_MAP)
_HAS_VECTORSTORE = Path(settings.CHROMA_PERSIST_DIR).exists()


def _fake_ensemble(retrievers, weights):
    return SimpleNamespace(retrievers=retrievers)


def _build(role: str, vectorstore: MagicMock | None = None):
    """Build the role's retriever without ChromaDB: a mock vector store, real BM25."""
    with patch.object(store, "get_vectorstore", return_value=vectorstore or MagicMock()), \
         patch.object(store, "EnsembleRetriever", _fake_ensemble):
        return store.get_rbac_retriever(ROLE_NAMESPACE_MAP[role])


def _bm25_half(role: str):
    return _build(role).retrievers[1]


@pytest.mark.parametrize("role", ROLES)
def test_bm25_index_holds_only_the_roles_namespaces(role):
    allowed = set(ROLE_NAMESPACE_MAP[role])

    indexed = {doc.metadata["namespace"] for doc in _bm25_half(role).docs}

    assert indexed == allowed  # nothing extra, and every allowed department is searchable


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("query", CROSS_DEPARTMENT_QUERIES)
def test_bm25_returns_no_chunks_from_other_departments(role, query):
    allowed = set(ROLE_NAMESPACE_MAP[role])

    docs = _bm25_half(role).invoke(query)

    assert docs
    assert {d.metadata["namespace"] for d in docs} <= allowed


@pytest.mark.parametrize("role", ROLES)
def test_vector_search_filters_by_the_roles_namespaces(role):
    vectorstore = MagicMock()
    _build(role, vectorstore)

    search_kwargs = vectorstore.as_retriever.call_args.kwargs["search_kwargs"]
    assert search_kwargs["filter"] == {"namespace": {"$in": ROLE_NAMESPACE_MAP[role]}}


@pytest.mark.skipif(not _HAS_VECTORSTORE, reason="ChromaDB not ingested")
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("query", CROSS_DEPARTMENT_QUERIES)
def test_full_retriever_returns_no_chunks_from_other_departments(role, query):
    allowed = set(ROLE_NAMESPACE_MAP[role])

    docs = store.get_rbac_retriever(ROLE_NAMESPACE_MAP[role]).invoke(query)

    assert docs
    assert {d.metadata["namespace"] for d in docs} <= allowed
