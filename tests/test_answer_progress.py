"""Eval answer collection survives rate limits: waits as long as the provider asks,
saves each answer as it arrives, and resumes where an interrupted run stopped."""
import json
from unittest.mock import patch

import pytest

import src.monitoring.ragas_eval as ragas_eval
from src.monitoring.answer_progress import load_progress, parse_retry_after, save_progress

QUESTIONS = [
    {"question": "q1?", "role": "General", "type": "factual"},
    {"question": "q2?", "role": "General", "type": "factual"},
    {"question": "q3?", "role": "General", "type": "factual"},
]


@pytest.mark.parametrize("message, seconds", [
    ("Please try again in 8m40.99s. Need more tokens?", 520.99),
    ("Please try again in 577ms.", 0.577),
    ("Please try again in 1h2m3s.", 3723.0),
    ("Please retry in 12.5s.", 12.5),  # Gemini wording
])
def test_parses_provider_retry_wait(message, seconds):
    assert parse_retry_after(message) == pytest.approx(seconds)


def test_unknown_wait_format_returns_none():
    assert parse_retry_after("Error code: 429 - slow down") is None


def test_retry_waits_as_long_as_groq_asks():
    error = Exception("Error code: 429 - Please try again in 2m0s.")
    with patch.object(ragas_eval, "run_query", side_effect=[error, {"answer": "ok"}]), \
         patch.object(ragas_eval.time, "sleep") as sleep:
        result = ragas_eval._run_query_with_retry("q?", ["general"], llm=None)

    assert result == {"answer": "ok"}
    sleep.assert_called_once_with(125.0)  # 120s asked + 5s margin


def test_progress_from_a_different_prompt_or_model_is_ignored(tmp_path):
    path = tmp_path / "progress.json"
    save_progress(path, "old-fingerprint", {"q1?": {"response": "a"}})

    assert load_progress(path, "new-fingerprint") == {}
    assert load_progress(path, "old-fingerprint") == {"q1?": {"response": "a"}}


def _record(item):
    q = item["question"]
    return {"response": f"answer to {q}", "contexts": [f"ctx for {q}"], "blocked": False}


def _collect(tmp_path, answer):
    with patch.object(ragas_eval, "_PROGRESS_FILE", tmp_path / "progress.json"), \
         patch.object(ragas_eval, "_CACHE_FILE", tmp_path / "cache.json"), \
         patch.object(ragas_eval, "load_eval_questions", return_value=QUESTIONS), \
         patch.object(ragas_eval, "_answer_question", side_effect=answer) as answer_mock, \
         patch.object(ragas_eval.time, "sleep"):
        result = ragas_eval._collect_samples(eval_only=False, subset=False)
    return result, answer_mock


def test_interrupted_run_keeps_answers_and_resume_asks_only_the_rest(tmp_path):
    def fail_on_third(item):
        if item["question"] == "q3?":
            raise RuntimeError("daily token cap")
        return _record(item)

    with pytest.raises(RuntimeError):
        _collect(tmp_path, fail_on_third)
    assert not (tmp_path / "cache.json").exists()

    (samples, n_total, skipped), answer_mock = _collect(tmp_path, _record)

    assert [c.args[0]["question"] for c in answer_mock.call_args_list] == ["q3?"]
    assert [s.user_input for s in samples] == ["q1?", "q2?", "q3?"]
    assert samples[0].response == "answer to q1?"
    assert (n_total, skipped) == (3, 0)
    assert not (tmp_path / "progress.json").exists()  # cleared once the cache is complete
    assert len(json.loads((tmp_path / "cache.json").read_text())) == 3


def test_blocked_questions_stay_skipped_on_resume(tmp_path):
    def block_q2(item):
        if item["question"] == "q2?":
            return {"response": "", "contexts": [], "blocked": True}
        if item["question"] == "q3?":
            raise RuntimeError("daily token cap")
        return _record(item)

    with pytest.raises(RuntimeError):
        _collect(tmp_path, block_q2)

    (samples, _, skipped), _ = _collect(tmp_path, _record)

    assert [s.user_input for s in samples] == ["q1?", "q3?"]
    assert skipped == 1
