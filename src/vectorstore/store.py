from functools import lru_cache

import structlog
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.config import settings

log = structlog.get_logger()


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name="company_docs",
        embedding_function=get_embeddings(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


@lru_cache(maxsize=1)
def _get_corpus() -> list[Document]:
    """Load and split all source documents once, cached for BM25 indexing.

    Deferred imports avoid the circular dependency:
    store → ingest → store (ingest imports get_vectorstore).
    By the time this is first called, all modules are fully loaded.
    """
    from src.ingestion.ingest import _apply_breadcrumbs
    from src.ingestion.loaders import load_all_documents
    from src.ingestion.splitter import get_splitter

    docs = load_all_documents()
    chunks = get_splitter().split_documents(docs)
    chunks = _apply_breadcrumbs(chunks)
    log.info("bm25_corpus_loaded", num_chunks=len(chunks))
    return chunks


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    log.info("cross_encoder_loading", model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_docs(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """Re-rank candidate docs with a cross-encoder; return top_k by descending score."""
    if not docs:
        return docs
    cross_encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def get_rbac_retriever(allowed_namespaces: list[str]):
    """Hybrid BM25 + vector ensemble retriever (weights: 0.6 vector, 0.4 BM25).

    The caller should apply rerank_docs() on the result of .invoke() for
    cross-encoder re-ranking before passing docs to the LLM.
    """
    fetch_k = settings.RETRIEVAL_FETCH_K

    vs = get_vectorstore()
    vector_retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": fetch_k,
            "fetch_k": fetch_k * 2,
            "filter": {"namespace": {"$in": allowed_namespaces}},
        },
    )

    corpus = _get_corpus()
    ns_set = set(allowed_namespaces)
    ns_chunks = [c for c in corpus if c.metadata.get("namespace") in ns_set]

    if not ns_chunks:
        log.warning("bm25_no_chunks_for_namespaces", namespaces=allowed_namespaces)
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(ns_chunks, k=fetch_k)

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )
