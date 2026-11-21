"""Helpers that let eval answer collection survive provider rate limits.

A full run can hit a daily token cap partway through. Each answer is saved to a
progress file as it arrives, so re-running resumes instead of starting over.
The file is tagged with a fingerprint of the prompt and model; answers produced
under a different prompt or model are never mixed into a resumed run.
"""
import hashlib
import json
import re
from pathlib import Path

# "try again in 8m40.99s" / "577ms" / "1h2m3s" (Groq), "retry in 12.5s" (Gemini)
_RETRY_RE = re.compile(r"(?:try again|retry) in ((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+(?:ms|s))?)")
_UNIT_RE = re.compile(r"([\d.]+)(ms|h|m|s)")
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def parse_retry_after(message: str) -> float | None:
    """Seconds the provider asks us to wait, or None if the message doesn't say."""
    match = _RETRY_RE.search(message)
    if not match or not match.group(1):
        return None
    return sum(float(n) * _UNIT_SECONDS[unit] for n, unit in _UNIT_RE.findall(match.group(1)))


def answer_fingerprint(prompt_template: str, model: str) -> str:
    """Identifies the setup that produced the answers: prompt text and RAG model."""
    return hashlib.sha256(f"{model}\n{prompt_template}".encode()).hexdigest()[:16]


def load_progress(path: Path, fingerprint: str) -> dict[str, dict]:
    """Saved answers keyed by question, or {} if none were saved under this fingerprint."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("fingerprint") != fingerprint:
        print(f"Ignoring {path.name}: it was made with a different prompt or model.")
        return {}
    return data["answers"]


def save_progress(path: Path, fingerprint: str, answers: dict[str, dict]) -> None:
    """Write all answers so far; via a temp file so an interruption can't corrupt it."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"fingerprint": fingerprint, "answers": answers}, indent=2), encoding="utf-8")
    tmp.replace(path)
