"""Tests for per-question score reports and their breakdown by type and role."""
import json
import math

import pandas as pd
import pytest

from src.monitoring.score_report import (
    build_per_question,
    group_scores,
    write_score_report,
)

NAN = float("nan")

META = {
    "q1": {"type": "factual", "role": "HR"},
    "q2": {"type": "factual", "role": "Finance"},
    "q3": {"type": "no_answer", "role": "HR"},
}


def _df(rows):
    return pd.DataFrame(rows)


def test_per_question_rows_carry_type_role_and_scores():
    df = _df([
        {"user_input": "q1", "faithfulness": 0.9, "answer_relevancy": 0.8},
        {"user_input": "q3", "faithfulness": NAN, "answer_relevancy": 0.1},
    ])

    rows = build_per_question(df, META)

    assert rows == [
        {"question": "q1", "type": "factual", "role": "HR",
         "faithfulness": 0.9, "answer_relevancy": 0.8},
        {"question": "q3", "type": "no_answer", "role": "HR",
         "faithfulness": None, "answer_relevancy": 0.1},
    ]


def test_question_missing_from_metadata_is_labelled_unknown():
    df = _df([{"user_input": "q-new", "faithfulness": 0.5, "answer_relevancy": 0.5}])

    rows = build_per_question(df, META)

    assert rows[0]["type"] == "unknown"
    assert rows[0]["role"] == "unknown"


def test_group_scores_averages_only_scored_samples():
    rows = [
        {"question": "q1", "type": "factual", "role": "HR", "faithfulness": 1.0, "answer_relevancy": 0.6},
        {"question": "q2", "type": "factual", "role": "Finance", "faithfulness": None, "answer_relevancy": 0.8},
        {"question": "q3", "type": "no_answer", "role": "HR", "faithfulness": 0.4, "answer_relevancy": 0.0},
    ]

    by_type = group_scores(rows, "type")

    assert by_type["factual"]["n"] == 2
    assert by_type["factual"]["faithfulness"] == pytest.approx(1.0)
    assert by_type["factual"]["n_scored"]["faithfulness"] == 1
    assert by_type["factual"]["answer_relevancy"] == pytest.approx(0.7)
    assert by_type["no_answer"]["answer_relevancy"] == pytest.approx(0.0)
    assert set(group_scores(rows, "role")) == {"HR", "Finance"}


def test_group_with_no_scored_samples_reports_none():
    rows = [{"question": "q", "type": "edge_case", "role": "HR", "faithfulness": None, "answer_relevancy": None}]

    group = group_scores(rows, "type")["edge_case"]

    assert group["faithfulness"] is None
    assert group["n_scored"] == {"faithfulness": 0, "answer_relevancy": 0}


def test_write_score_report_saves_judge_rows_and_breakdowns(tmp_path):
    df = _df([
        {"user_input": "q1", "faithfulness": 0.9, "answer_relevancy": 0.8},
        {"user_input": "q3", "faithfulness": 0.2, "answer_relevancy": 0.1},
    ])
    path = tmp_path / "scores.json"

    report = write_score_report(df, META, path, judge_provider="deepseek", judge_model="deepseek-chat")

    saved = json.loads(path.read_text())
    assert saved == report
    assert saved["judge_provider"] == "deepseek"
    assert saved["judge_model"] == "deepseek-chat"
    assert len(saved["per_question"]) == 2
    assert set(saved["by_type"]) == {"factual", "no_answer"}
    assert set(saved["by_role"]) == {"HR"}
    assert "timestamp" in saved


def test_saved_report_is_strict_json_without_nan(tmp_path):
    df = _df([{"user_input": "q1", "faithfulness": NAN, "answer_relevancy": NAN}])
    path = tmp_path / "scores.json"

    write_score_report(df, META, path, judge_provider="groq", judge_model="m")

    json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
