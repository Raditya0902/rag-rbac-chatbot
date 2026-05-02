import csv
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

DATA_DIR = Path("resources/data")

DEPARTMENT_DIRS = {
    "engineering": DATA_DIR / "engineering",
    "finance":     DATA_DIR / "finance",
    "general":     DATA_DIR / "general",
    "hr":          DATA_DIR / "hr",
    "marketing":   DATA_DIR / "marketing",
}

_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


def file_md5(path: Path) -> str:
    """Return the MD5 hex digest of a file's contents."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_markdown_docs() -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_MD_HEADERS, strip_headers=False)
    docs = []
    for namespace, dir_path in DEPARTMENT_DIRS.items():
        if not dir_path.exists():
            continue
        for md_file in dir_path.glob("*.md"):
            doc_version = file_md5(md_file)
            text = md_file.read_text(encoding="utf-8")
            sections = splitter.split_text(text)
            for section in sections:
                section.metadata["namespace"] = namespace
                section.metadata["source"] = md_file.name
                section.metadata["doc_version"] = doc_version
                docs.append(section)
    return docs


def load_csv_as_sentences(csv_path: Path, namespace: str) -> list[Document]:
    doc_version = file_md5(csv_path)
    docs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parts = [
                f"{key.replace('_', ' ').title()}: {val.strip()}"
                for key, val in row.items()
                if val and val.strip()
            ]
            if parts:
                docs.append(Document(
                    page_content=". ".join(parts) + ".",
                    metadata={
                        "namespace": namespace,
                        "source": csv_path.name,
                        "doc_version": doc_version,
                    },
                ))
    return docs


def load_all_documents() -> list[Document]:
    docs = load_markdown_docs()
    hr_csv = DATA_DIR / "hr" / "hr_data.csv"
    if hr_csv.exists():
        docs.extend(load_csv_as_sentences(hr_csv, "hr"))
    return docs
