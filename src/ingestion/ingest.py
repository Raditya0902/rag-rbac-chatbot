from langchain.schema import Document

from src.ingestion.loaders import load_all_documents
from src.ingestion.splitter import get_splitter
from src.vectorstore.store import get_vectorstore

_HEADER_KEYS = ("h1", "h2", "h3", "h4")


def _apply_breadcrumbs(chunks: list[Document]) -> list[Document]:
    """Prepend section breadcrumb to every chunk so all sub-chunks carry header context."""
    for chunk in chunks:
        breadcrumb = " > ".join(
            chunk.metadata[k] for k in _HEADER_KEYS if chunk.metadata.get(k)
        )
        if breadcrumb and not chunk.page_content.startswith("["):
            chunk.page_content = f"[{breadcrumb}]\n{chunk.page_content}"
    return chunks


def ingest() -> None:
    print("Loading documents...")
    docs = load_all_documents()
    print(f"Loaded {len(docs)} raw documents")

    chunks = get_splitter().split_documents(docs)
    chunks = _apply_breadcrumbs(chunks)
    print(f"Split into {len(chunks)} chunks")

    namespace_counts: dict[str, int] = {}
    for c in chunks:
        ns = c.metadata.get("namespace", "unknown")
        namespace_counts[ns] = namespace_counts.get(ns, 0) + 1
    for ns, count in sorted(namespace_counts.items()):
        print(f"  {ns}: {count} chunks")

    vs = get_vectorstore()
    vs.reset_collection()
    vs.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks into ChromaDB")


if __name__ == "__main__":
    ingest()
