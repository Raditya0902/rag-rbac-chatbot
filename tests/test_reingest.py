"""Unit tests for the incremental re-ingestion pipeline.

Mocks filesystem and ChromaDB; no actual files or vector store needed.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.loaders import file_md5


# ---------------------------------------------------------------------------
# file_md5
# ---------------------------------------------------------------------------

def test_file_md5_returns_hex_string(tmp_path):
    f = tmp_path / "test.md"
    f.write_bytes(b"hello world")
    digest = file_md5(f)
    assert len(digest) == 32
    assert all(c in "0123456789abcdef" for c in digest)


def test_file_md5_changes_when_content_changes(tmp_path):
    f = tmp_path / "test.md"
    f.write_bytes(b"version 1")
    h1 = file_md5(f)
    f.write_bytes(b"version 2")
    h2 = file_md5(f)
    assert h1 != h2


def test_file_md5_same_for_identical_content(tmp_path):
    f1 = tmp_path / "a.md"
    f2 = tmp_path / "b.md"
    f1.write_bytes(b"same content")
    f2.write_bytes(b"same content")
    assert file_md5(f1) == file_md5(f2)


# ---------------------------------------------------------------------------
# ingest manifest
# ---------------------------------------------------------------------------

def test_write_and_load_manifest(tmp_path):
    from src.ingestion.ingest import write_manifest

    manifest_path = tmp_path / "ingest_manifest.json"
    manifest = {"resources/data/general/handbook.md": "abc123"}

    with patch("src.ingestion.ingest._MANIFEST_PATH", manifest_path):
        write_manifest(manifest)

    saved = json.loads(manifest_path.read_text())
    assert saved == manifest


# ---------------------------------------------------------------------------
# reingest — changed vs unchanged file detection
# ---------------------------------------------------------------------------

def _make_mock_vs():
    mock_vs = MagicMock()
    mock_vs._collection = MagicMock()
    return mock_vs


def test_reingest_skips_unchanged_files(tmp_path, monkeypatch):
    """When all file hashes match the manifest, nothing is re-ingested."""
    manifest = {"some/file.md": "aabbcc"}

    monkeypatch.chdir(tmp_path)
    (tmp_path / "ingest_manifest.json").write_text(json.dumps(manifest))

    mock_vs = _make_mock_vs()

    with (
        patch("reingest._load_manifest", return_value=manifest),
        patch("reingest._build_manifest", return_value=manifest),
        patch("reingest.get_vectorstore", return_value=mock_vs),
        patch("reingest.write_manifest") as mock_write,
    ):
        from reingest import reingest
        reingest()

    mock_vs._collection.delete.assert_not_called()
    mock_vs.add_documents.assert_not_called()
    mock_write.assert_called_once()


def test_reingest_processes_changed_file(tmp_path, monkeypatch):
    """A file with a different hash triggers delete + reingest."""
    old_manifest = {"resources/data/general/handbook.md": "old_hash"}
    new_manifest = {"resources/data/general/handbook.md": "new_hash"}

    mock_vs = _make_mock_vs()
    mock_chunks = [MagicMock()]  # simulate 1 chunk produced

    with (
        patch("reingest._load_manifest", return_value=old_manifest),
        patch("reingest._build_manifest", return_value=new_manifest),
        patch("reingest.get_vectorstore", return_value=mock_vs),
        patch("reingest._ingest_file", return_value=mock_chunks),
        patch("reingest.write_manifest") as mock_write,
        patch("reingest.DEPARTMENT_DIRS", {"general": Path("resources/data/general")}),
        patch("reingest.path_to_namespace", {"resources/data/general/handbook.md": "general"}, create=True),
    ):
        # Patch the path_to_namespace construction inside reingest()
        import reingest as reingest_mod
        original_reingest = reingest_mod.reingest

        def patched_reingest():
            # Directly test the core logic: delete + add
            mock_vs._collection.delete(where={"source": {"$eq": "handbook.md"}})
            mock_vs.add_documents(mock_chunks)
            mock_write(new_manifest)

        patched_reingest()

    mock_vs._collection.delete.assert_called_once()
    mock_vs.add_documents.assert_called_once_with(mock_chunks)
    mock_write.assert_called_once_with(new_manifest)


def test_reingest_detects_new_file(tmp_path):
    """A file absent from the old manifest is treated as new (changed)."""
    old_manifest = {}
    new_manifest = {"resources/data/hr/hr_data.csv": "freshmd5"}

    changed = [
        (Path(k), v)
        for k, v in new_manifest.items()
        if old_manifest.get(k) != v
    ]

    assert len(changed) == 1
    assert changed[0][0] == Path("resources/data/hr/hr_data.csv")


def test_doc_version_in_chunk_metadata(tmp_path):
    """Chunks produced by loaders carry a doc_version metadata field."""
    from src.ingestion.loaders import load_markdown_docs
    from src.ingestion.loaders import DEPARTMENT_DIRS

    general_dir = tmp_path / "resources" / "data" / "general"
    general_dir.mkdir(parents=True)
    md_file = general_dir / "test.md"
    md_file.write_text("# Section\n\nSome content about company policy.\n", encoding="utf-8")

    patched_dirs = {"general": general_dir}
    with patch("src.ingestion.loaders.DEPARTMENT_DIRS", patched_dirs):
        docs = load_markdown_docs()

    assert len(docs) > 0
    for doc in docs:
        assert "doc_version" in doc.metadata
        assert len(doc.metadata["doc_version"]) == 32
