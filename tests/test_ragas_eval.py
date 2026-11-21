"""Tests for RAGAS score aggregation and reporting.

Offline: RAGAS `evaluate` is replaced with a stub that returns a fixed score
table, so no LLM or network calls are made.
"""
import json
import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import src.monitoring.ragas_eval as ragas_eval


def _write_cache(path, n):
    samples = [
        {"user_input": f"q{i}", "response": f"a{i}", "retrieved_contexts": [f"c{i}"]}
        for i in range(n)
    ]
    path.write_text(json.dumps(samples))


def _run_with_scores(tmp_path, rows, record_note=None):
    cache = tmp_path / "cache.json"
    results = tmp_path / "EVAL_RESULTS.md"
    results.write_text("# log\n")
    _write_cache(cache, len(rows))
    fake_result = MagicMock()
    fake_result.to_pandas.return_value = pd.DataFrame(rows)
    with patch.object(ragas_eval, "_CACHE_FILE", cache), \
         patch.object(ragas_eval, "_RESULTS_FILE", results), \
         patch.object(ragas_eval, "ChatGroq", MagicMock()), \
         patch.object(ragas_eval, "get_embeddings", MagicMock()), \
         patch.object(ragas_eval, "evaluate", return_value=fake_result):
        out = ragas_eval.run_evaluation(record_note=record_note, eval_only=True)
    return out, results.read_text()


NAN = float("nan")


def test_reports_scored_counts_when_some_samples_fail(tmp_path):
    rows = [
        {"faithfulness": 1.0, "answer_relevancy": 0.5},
        {"faithfulness": NAN, "answer_relevancy": 0.7},
        {"faithfulness": NAN, "answer_relevancy": NAN},
    ]

    out, _ = _run_with_scores(tmp_path, rows)

    assert out["n_questions"] == 3
    assert out["n_scored"] == {"faithfulness": 1, "answer_relevancy": 2}
    assert out["faithfulness"] == pytest.approx(1.0)
    assert out["answer_relevancy"] == pytest.approx(0.6)


def test_does_not_record_run_with_unscored_samples(tmp_path):
    rows = [
        {"faithfulness": 1.0, "answer_relevancy": 0.5},
        {"faithfulness": NAN, "answer_relevancy": 0.7},
    ]

    _, log = _run_with_scores(tmp_path, rows, record_note="partial failure")

    assert "partial failure" not in log


def test_does_not_record_run_where_every_sample_failed(tmp_path):
    rows = [{"faithfulness": NAN, "answer_relevancy": NAN}] * 2

    out, log = _run_with_scores(tmp_path, rows, record_note="all failed")

    assert math.isnan(out["faithfulness"])
    assert out["n_scored"] == {"faithfulness": 0, "answer_relevancy": 0}
    assert "all failed" not in log


def test_records_run_when_every_sample_scored(tmp_path):
    rows = [
        {"faithfulness": 0.8, "answer_relevancy": 0.6},
        {"faithfulness": 1.0, "answer_relevancy": 0.8},
    ]

    _, log = _run_with_scores(tmp_path, rows, record_note="clean run")

    assert "clean run" in log
    assert "faithfulness:     0.9000" in log
    assert "questions:        2" in log
