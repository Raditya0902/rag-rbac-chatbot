import argparse
import json
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
from src.monitoring.judge import build_judge_chat_model, resolve_judge
from src.rag.chain import run_query
from src.vectorstore.store import get_embeddings, get_rbac_retriever, rerank_docs

_QUESTIONS_FILE = Path(__file__).parents[2] / "eval_questions.json"
_SUBSET_FILE = Path(__file__).parents[2] / "eval_questions_subset.json"
_CACHE_FILE = Path(__file__).parents[2] / "eval_samples_cache.json"
_RESULTS_FILE = Path(__file__).parents[2] / "EVAL_RESULTS.md"
_METRIC_NAMES = ("faithfulness", "answer_relevancy")


def summarize_scores(df: pd.DataFrame) -> dict:
    """Mean of each metric over the samples RAGAS actually scored.

    RAGAS returns NaN for a sample when the judge LLM's output can't be parsed
    or the call fails. Those rows are excluded from the mean and counted
    separately, so a partial run is never mistaken for a complete one.
    """
    return {
        "means": {m: float(df[m].mean()) for m in _METRIC_NAMES},
        "n_scored": {m: int(df[m].notna().sum()) for m in _METRIC_NAMES},
    }


def load_eval_questions(subset: bool = False) -> list[dict]:
    path = _SUBSET_FILE if (subset and _SUBSET_FILE.exists()) else _QUESTIONS_FILE
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_query_with_retry(question: str, namespaces: list, llm, max_wait: int = 3600) -> dict:
    """Call run_query with the given LLM, retrying on rate-limit errors."""
    import re

    waited = 0
    while True:
        try:
            return run_query(question, namespaces, llm=llm)
        except Exception as exc:
            msg = str(exc)
            if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg:
                raise
            match = re.search(r"retry in ([\d.]+)s", msg)
            secs = float(match.group(1)) + 5 if match else 60
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


def run_evaluation(
    record_note: str | None = None,
    subset: bool = False,
    eval_only: bool = False,
    judge_model: str | None = None,
    judge_provider: str | None = None,
) -> dict:
    judge_provider, judge_model = resolve_judge(judge_provider, judge_model)
    print(f"Judge: {judge_provider}/{judge_model}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        eval_llm = LangchainLLMWrapper(build_judge_chat_model(judge_provider, judge_model))
        eval_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    if eval_only:
        if not _CACHE_FILE.exists():
            raise FileNotFoundError(
                f"No cache at {_CACHE_FILE}. Run without --eval-only first to build it."
            )
        print(f"Loading cached samples from {_CACHE_FILE} ...")
        samples = _load_cache()
        skipped = 0
        n_total = len(samples)
    else:
        eval_questions = load_eval_questions(subset=subset)
        n_total = len(eval_questions)
        samples = []
        skipped = 0
        for item in eval_questions:
            namespaces = ROLE_NAMESPACE_MAP.get(item["role"], ["general"])
            result = _run_query_with_retry(item["question"], namespaces, llm=None)
            if result["guardrail_blocked"]:
                skipped += 1
                continue

            retriever = get_rbac_retriever(namespaces)
            candidates = retriever.invoke(item["question"])
            docs = rerank_docs(item["question"], candidates, settings.RETRIEVAL_TOP_K)
            retrieved_contexts = [doc.page_content for doc in docs]

            samples.append(SingleTurnSample(
                user_input=item["question"],
                response=result["answer"],
                retrieved_contexts=retrieved_contexts,
            ))
            time.sleep(2)

        _save_cache(samples)

    if not samples:
        print("No samples to evaluate (all blocked by guardrails).")
        return {}

    metrics = [
        Faithfulness(llm=eval_llm),
        AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings, strictness=1),
    ]

    scores = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        batch_size=5,
    )
    summary = summarize_scores(scores.to_pandas())
    means, n_scored = summary["means"], summary["n_scored"]
    n_samples = len(samples)
    is_complete = all(n == n_samples for n in n_scored.values())

    print("\n=== RAGAS Evaluation Results ===")
    print(f"  questions evaluated : {n_samples} / {n_total} ({skipped} skipped)")
    for name in _METRIC_NAMES:
        print(f"  {name:<20}: {means[name]:.4f}  (scored {n_scored[name]}/{n_samples})")
    print("================================\n")

    result_data = {
        "faithfulness": means["faithfulness"],
        "answer_relevancy": means["answer_relevancy"],
        "n_questions": n_samples,
        "n_scored": n_scored,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
    }

    if record_note is None:
        return result_data
    if not is_complete:
        print(
            "Not recording to EVAL_RESULTS.md: some samples failed to score "
            "(judge LLM error or rate limit). Re-run with --eval-only to retry."
        )
        return result_data
    _append_eval_results(
        means["faithfulness"], means["answer_relevancy"], n_samples,
        judge_provider, judge_model, record_note,
    )
    return result_data


def _append_eval_results(
    faithfulness: float,
    answer_relevancy: float,
    n_questions: int,
    judge_provider: str,
    judge_model: str,
    note: str,
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = (
        f"\n## {timestamp} — {note}\n"
        f"- faithfulness:     {faithfulness:.4f}\n"
        f"- answer_relevancy: {answer_relevancy:.4f}\n"
        f"- questions:        {n_questions}\n"
        f"- judge_provider:   {judge_provider}\n"
        f"- judge_model:      {judge_model}\n"
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
