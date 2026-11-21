"""Per-question RAGAS scores, grouped by question type and role.

Aggregate means hide where the pipeline is weak (e.g. a correct "I don't know"
scores poorly on answer relevancy). This module keeps every question's scores
next to its type and role so a run can be broken down after the fact.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

METRIC_NAMES = ("faithfulness", "answer_relevancy")
_UNKNOWN = "unknown"


def _score_or_none(value) -> float | None:
    """RAGAS marks unscored samples as NaN; store them as JSON null."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def build_per_question(df: pd.DataFrame, question_meta: dict[str, dict]) -> list[dict]:
    """One row per scored sample, joined to its type and role by question text."""
    rows = []
    for record in df.to_dict("records"):
        question = record["user_input"]
        meta = question_meta.get(question, {})
        rows.append({
            "question": question,
            "type": meta.get("type", _UNKNOWN),
            "role": meta.get("role", _UNKNOWN),
            **{m: _score_or_none(record.get(m)) for m in METRIC_NAMES},
        })
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def group_scores(rows: list[dict], key: str) -> dict[str, dict]:
    """Mean of each metric per group, over the samples that were actually scored."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)

    summary = {}
    for name, members in sorted(groups.items()):
        scored = {m: [r[m] for r in members if r[m] is not None] for m in METRIC_NAMES}
        summary[name] = {
            "n": len(members),
            **{m: _mean(scored[m]) for m in METRIC_NAMES},
            "n_scored": {m: len(scored[m]) for m in METRIC_NAMES},
        }
    return summary


def write_score_report(
    df: pd.DataFrame,
    question_meta: dict[str, dict],
    path: Path,
    judge_provider: str,
    judge_model: str,
    refusal: dict | None = None,
) -> dict:
    """Build the per-question report, save it as JSON, and return it.

    `per_question` and the breakdowns cover RAGAS-scored questions; `refusal`
    holds the separate refusal scoring of no_answer questions.
    """
    per_question = build_per_question(df, question_meta)
    report = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "by_type": group_scores(per_question, "type"),
        "by_role": group_scores(per_question, "role"),
        "refusal": refusal,
        "per_question": per_question,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return report


def _fmt(value: float | None) -> str:
    return "  n/a " if value is None else f"{value:.4f}"


def format_breakdown(groups: dict[str, dict], label: str) -> str:
    """Render one breakdown as a fixed-width text table for the console."""
    lines = [f"  {label:<12} {'n':>3}  {'faithful':>8}  {'relevancy':>9}"]
    for name, g in groups.items():
        lines.append(
            f"  {name:<12} {g['n']:>3}  {_fmt(g['faithfulness']):>8}  {_fmt(g['answer_relevancy']):>9}"
        )
    return "\n".join(lines)
