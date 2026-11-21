"""Guards against source files being excluded from version control.

A `.gitignore` rule once matched `src/vectorstore/`, so a fresh clone could not
import the RAG chain or the eval pipeline.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_no_source_file_is_gitignored():
    sources = sorted(str(p.relative_to(_REPO_ROOT)) for p in (_REPO_ROOT / "src").rglob("*.py"))

    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *sources],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )

    assert result.stdout.strip() == "", f"gitignored source files:\n{result.stdout}"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_lockfile_is_not_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "uv.lock"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )

    assert result.stdout.strip() == ""
