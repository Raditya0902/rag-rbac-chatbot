"""Incremental re-ingestion script.

Usage:
    python reingest.py

Behaviour:
- Hashes all source files in resources/data/.
- Compares against the stored manifest (ingest_manifest.json).
- For each file whose hash changed or that is new:
    1. Deletes existing ChromaDB chunks whose `source` metadata matches the filename.
    2. Re-ingests only those chunks (with updated doc_version metadata).
- Updates ingest_manifest.json with the new hashes.
- Files that are unchanged are skipped entirely.

The full `ingest.py` script (which wipes and rebuilds everything) should be used
for first-time setup. Do NOT run both scripts simultaneously.
"""
import json
from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.ingest import _apply_breadcrumbs, _build_manifest, write_manifest
from src.ingestion.loaders import (
    DEPARTMENT_DIRS,
    file_md5,
    load_csv_as_sentences,
    load_markdown_docs,
)
from src.ingestion.splitter import get_splitter
from src.vectorstore.store import get_vectorstore

_MANIFEST_PATH = Path("ingest_manifest.json")


def _load_manifest() -> dict[str, str]:
    if _MANIFEST_PATH.exists():
        return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def _delete_chunks_for_source(vs, filename: str) -> None:
    """Delete all ChromaDB chunks whose 'source' metadata equals filename."""
    try:
        vs._collection.delete(where={"source": {"$eq": filename}})
    except Exception as exc:
        print(f"  WARNING: could not delete chunks for {filename}: {exc}")


def _ingest_file(path: Path, namespace: str) -> list[Document]:
    """Load, split, and apply breadcrumbs for a single source file."""
    if path.suffix == ".md":
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        _MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_MD_HEADERS, strip_headers=False
        )
        doc_version = file_md5(path)
        sections = header_splitter.split_text(path.read_text(encoding="utf-8"))
        for s in sections:
            s.metadata["namespace"] = namespace
            s.metadata["source"] = path.name
            s.metadata["doc_version"] = doc_version
        docs = sections
    elif path.suffix == ".csv":
        docs = load_csv_as_sentences(path, namespace)
    else:
        return []

    chunks = get_splitter().split_documents(docs)
    chunks = _apply_breadcrumbs(chunks)
    return chunks


def reingest() -> None:
    old_manifest = _load_manifest()
    new_manifest = _build_manifest()

    changed: list[tuple[Path, str]] = []
    for file_str, new_hash in new_manifest.items():
        if old_manifest.get(file_str) != new_hash:
            changed.append((Path(file_str), new_hash))

    if not changed:
        print("No files changed. Nothing to re-ingest.")
        write_manifest(new_manifest)
        return

    print(f"{len(changed)} file(s) changed or new:")
    for path, _ in changed:
        status = "NEW" if str(path) not in old_manifest else "CHANGED"
        print(f"  [{status}] {path}")

    # Map each file path back to its namespace
    path_to_namespace: dict[str, str] = {}
    for namespace, dir_path in DEPARTMENT_DIRS.items():
        if dir_path.exists():
            for f in dir_path.iterdir():
                if f.suffix in {".md", ".csv"}:
                    path_to_namespace[str(f)] = namespace

    vs = get_vectorstore()
    total_chunks = 0

    for path, _ in changed:
        namespace = path_to_namespace.get(str(path), "general")
        print(f"\nProcessing: {path.name} (namespace={namespace})")

        # Remove stale chunks
        _delete_chunks_for_source(vs, path.name)
        print(f"  Deleted existing chunks for source='{path.name}'")

        # Re-ingest
        chunks = _ingest_file(path, namespace)
        if chunks:
            vs.add_documents(chunks)
            print(f"  Ingested {len(chunks)} new chunks")
            total_chunks += len(chunks)
        else:
            print(f"  No chunks produced (file may be empty)")

    write_manifest(new_manifest)
    print(f"\nDone. {total_chunks} chunks ingested across {len(changed)} file(s).")
    print(f"Manifest updated: {_MANIFEST_PATH}")


if __name__ == "__main__":
    reingest()
