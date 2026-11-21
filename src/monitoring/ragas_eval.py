import argparse
import json
import math
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy

from src.auth.rbac import ROLE_NAMESPACE_MAP
from src.config import settings
from src.monitoring.answer_progress import (
    answer_fingerprint, load_progress, parse_retry_after, save_progress,
)
from src.monitoring.judge import build_judge_chat_model, resolve_judge
from src.monitoring.refusal import split_by_answerability, summarize_refusals
from src.monitoring.score_report import METRIC_NAMES, format_breakdown, write_score_report
from src.rag.chain import run_query
from src.rag.prompts import RAG_PROMPT
from src.vectorstore.store import get_embeddings, get_rbac_retriever, rerank_docs

_QUESTIONS_FILE = Path(__file__).parents[2] / "eval_questions.json"
_SUBSET_FILE = Path(__file__).parents[2] / "eval_questions_subset.json"
_CACHE_FILE = Path(__file__).parents[2] / "eval_samples_cache.json"
_PROGRESS_FILE = Path(__file__).parents[2] / "eval_samples_progress.json"
_RETRY_MARGIN_S = 5
_DEFAULT_RETRY_WAIT_S = 60
_RESULTS_FILE = Path(__file__).parents[2] / "EVAL_RESULTS.md"
_SCORES_FILE = Path(__file__).parents[2] / "eval_scores.json"


def summarize_scores(df: pd.DataFrame) -> dict:
    """Mean of each metric over the samples RAGAS actually scored.

    RAGAS returns NaN for a sample when the judge LLM's output can't be parsed
    or the call fails. Those rows are excluded from the mean and counted
    separately, so a partial run is never mistaken for a complete one.
    """
    return {
        "means": {m: float(df[m].mean()) if len(df) else None for m in METRIC_NAMES},
        "n_scored": {m: int(df[m].notna().sum()) for m in METRIC_NAMES},
    }


def _load_question_meta() -> dict[str, dict]:
    """Map each eval question's text to its type and role."""
    with open(_QUESTIONS_FILE, encoding="utf-8") as f:
        return {q["question"]: {"type": q["type"], "role": q["role"]} for q in json.load(f)}


def load_eval_questions(subset: bool = False) -> list[dict]:
    path = _SUBSET_FILE if (subset and _SUBSET_FILE.exists()) else _QUESTIONS_FILE
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_query_with_retry(question: str, namespaces: list, llm, max_wait: int = 3600) -> dict:
    """Call run_query with the given LLM, waiting out rate limits as long as the provider asks."""
    waited = 0.0
    while True:
        try:
            return run_query(question, namespaces, llm=llm)
        except Exception as exc:
            msg = str(exc)
            if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg:
                raise
            asked = parse_retry_after(msg)
            secs = (asked if asked is not None else _DEFAULT_RETRY_WAIT_S) + _RETRY_MARGIN_S
            if waited + secs > max_wait:
                raise
            print(f"  Rate limited — waiting {secs:.0f}s (total waited: {waited + secs:.0f}s)...")
            time.sleep(secs)
            waited += secs


def _save_cache(samples: list[SingleTurnSample]) -> None:
    data = [
        {
            "user_input": s.user_input,
            "response": s.response,
            "retrieved_contexts": s.retrieved_contexts,
        }
        for s in samples
    ]
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Cached {len(data)} samples → {_CACHE_FILE}")


def _load_cache() -> list[SingleTurnSample]:
    with open(_CACHE_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return [
        SingleTurnSample(
            user_input=d["user_input"],
            response=d["response"],
            retrieved_contexts=d["retrieved_contexts"],
        )
        for d in data
    ]


def _answer_question(item: dict) -> dict:
    """Ask the RAG chain one eval question; return its answer and the contexts it used."""
    namespaces = ROLE_NAMESPACE_MAP.get(item["role"], ["general"])
    result = _run_query_with_retry(item["question"], namespaces, llm=None)
    if result["guardrail_blocked"]:
        return {"response": "", "contexts": [], "blocked": True}
    candidates = get_rbac_retriever(namespaces).invoke(item["question"])
    docs = rerank_docs(item["question"], candidates, settings.RETRIEVAL_TOP_K)
    return {"response": result["answer"], "contexts": [d.page_content for d in docs], "blocked": False}


def _collect_samples(eval_only: bool, subset: bool) -> tuple[list[SingleTurnSample], int, int]:
    """Return (samples, n_total, n_guardrail_skipped), from cache or by querying the RAG chain.

    Answers are saved to _PROGRESS_FILE as they arrive; an interrupted run resumes from it.
    """
    if eval_only:
        if not _CACHE_FILE.exists():
            raise FileNotFoundError(
                f"No cache at {_CACHE_FILE}. Run without --eval-only first to build it."
            )
        print(f"Loading cached samples from {_CACHE_FILE} ...")
        samples = _load_cache()
        return samples, len(samples), 0

    eval_questions = load_eval_questions(subset=subset)
    fingerprint = answer_fingerprint(RAG_PROMPT.messages[0].prompt.template, settings.LLM_MODEL)
    answers = load_progress(_PROGRESS_FILE, fingerprint)
    if answers:
        print(f"Resuming: {len(answers)}/{len(eval_questions)} answers already saved in {_PROGRESS_FILE.name}")

    for item in eval_questions:
        if item["question"] in answers:
            continue
        try:
            record = _answer_question(item)
        except Exception:
            print(f"Stopped at {len(answers)}/{len(eval_questions)} answers; "
                  f"they are saved in {_PROGRESS_FILE.name}. Re-run to resume.")
            raise
        answers = {**answers, item["question"]: record}
        save_progress(_PROGRESS_FILE, fingerprint, answers)
        time.sleep(2)

    records = [(item["question"], answers[item["question"]]) for item in eval_questions]
    samples = [
        SingleTurnSample(user_input=q, response=r["response"], retrieved_contexts=r["contexts"])
        for q, r in records if not r["blocked"]
    ]
    _save_cache(samples)
    _PROGRESS_FILE.unlink(missing_ok=True)
    return samples, len(eval_questions), len(records) - len(samples)


def _score_with_ragas(samples: list[SingleTurnSample], judge_provider: str, judge_model: str) -> pd.DataFrame:
    """Score samples with RAGAS faithfulness and answer relevancy; one row per sample."""
    if not samples:
        return pd.DataFrame(columns=["user_input", *METRIC_NAMES])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        eval_llm = LangchainLLMWrapper(build_judge_chat_model(judge_provider, judge_model))
        eval_embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    scores = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(llm=eval_llm),
            AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings, strictness=1),
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
        batch_size=5,
    )
    return scores.to_pandas()


def _fmt_ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}"


def _print_results(result: dict, n_total: int, skipped: int) -> None:
    refusal = result["refusal"]
    n_answerable = refusal["n_answerable"]
    print("\n=== RAGAS Evaluation Results ===")
    print(f"  questions evaluated : {result['n_questions']} / {n_total} ({skipped} skipped)")
    print(f"  RAGAS (answerable)  : {n_answerable} questions")
    for name in METRIC_NAMES:
        value = result[name]
        shown = "n/a" if value is None or math.isnan(value) else f"{value:.4f}"
        print(f"    {name:<18}: {shown}  (scored {result['n_scored'][name]}/{n_answerable})")
    print(f"  refusal accuracy    : {_fmt_ratio(refusal['correct'], refusal['n_no_answer'])} no_answer questions refused")
    print(f"  false refusals      : {_fmt_ratio(refusal['false_refusals'], n_answerable)} answerable questions refused")
    for question in refusal["missed"]:
        print(f"    missed refusal: {question[:80]}")
    for question in refusal["false_refusal_questions"]:
        print(f"    false refusal : {question[:80]}")
    print("================================\n")


def run_evaluation(
    record_note: str | None = None,
    subset: bool = False,
    eval_only: bool = False,
    judge_model: str | None = None,
    judge_provider: str | None = None,
) -> dict:
    judge_provider, judge_model = resolve_judge(judge_provider, judge_model)
    print(f"Judge: {judge_provider}/{judge_model}")

    samples, n_total, skipped = _collect_samples(eval_only, subset)
    if not samples:
        print("No samples to evaluate (all blocked by guardrails).")
        return {}

    question_meta = _load_question_meta()
    answerable, no_answer = split_by_answerability(samples, question_meta)
    refusal = summarize_refusals(answerable, no_answer)
    df = _score_with_ragas(answerable, judge_provider, judge_model)
    summary = summarize_scores(df)

    result_data = {
        "faithfulness": summary["means"]["faithfulness"],
        "answer_relevancy": summary["means"]["answer_relevancy"],
        "n_questions": len(samples),
        "n_scored": summary["n_scored"],
        "refusal": refusal,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
    }
    _print_results(result_data, n_total, skipped)

    report = write_score_report(
        df, question_meta, _SCORES_FILE,
        judge_provider=judge_provider, judge_model=judge_model, refusal=refusal,
    )
    print(format_breakdown(report["by_type"], "type"))
    print()
    print(format_breakdown(report["by_role"], "role"))
    print(f"\nPer-question scores saved → {_SCORES_FILE}\n")

    if record_note is None:
        return result_data
    if not answerable:
        print("Not recording to EVAL_RESULTS.md: no answerable questions were scored by RAGAS.")
        return result_data
    if not all(n == len(answerable) for n in summary["n_scored"].values()):
        print(
            "Not recording to EVAL_RESULTS.md: some samples failed to score "
            "(judge LLM error or rate limit). Re-run with --eval-only to retry."
        )
        return result_data
    _append_eval_results(result_data, record_note)
    return result_data


def _append_eval_results(result: dict, note: str) -> None:
    refusal = result["refusal"]
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = (
        f"\n## {timestamp} — {note}\n"
        f"- faithfulness:     {result['faithfulness']:.4f}\n"
        f"- answer_relevancy: {result['answer_relevancy']:.4f}\n"
        f"- questions:        {result['n_questions']} "
        f"({refusal['n_answerable']} answerable scored by RAGAS, {refusal['n_no_answer']} no_answer)\n"
        f"- refusal_accuracy: {_fmt_ratio(refusal['correct'], refusal['n_no_answer'])}\n"
        f"- false_refusals:   {_fmt_ratio(refusal['false_refusals'], refusal['n_answerable'])}\n"
        f"- judge_provider:   {result['judge_provider']}\n"
        f"- judge_model:      {result['judge_model']}\n"
        f"- note: {note}\n"
    )
    with open(_RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"Results appended to {_RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--record",
        metavar="NOTE",
        default=None,
        help="Append results to EVAL_RESULTS.md with this note",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        default=False,
        help="Use eval_questions_subset.json instead of the full 50-question set",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Skip RAG queries; score from cached eval_samples_cache.json",
    )
    parser.add_argument(
        "--judge-model",
        metavar="MODEL",
        default=None,
        help="Judge model for RAGAS scoring (default: EVAL_JUDGE_MODEL, else the provider's default)",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["groq", "deepseek"],
        default=None,
        help="Judge LLM provider (default: EVAL_JUDGE_PROVIDER setting)",
    )
    args = parser.parse_args()
    run_evaluation(
        record_note=args.record,
        subset=args.subset,
        eval_only=args.eval_only,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
    )
