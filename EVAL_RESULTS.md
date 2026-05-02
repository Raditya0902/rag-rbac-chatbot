# RAGAS Evaluation Results

Append-only log of evaluation runs. Each entry is produced by:

```
python -m src.monitoring.ragas_eval --record "description of what changed"
```

Metrics:
- **faithfulness**: How well the answer is grounded in the retrieved context (0–1, higher is better)
- **answer_relevancy**: How relevant the answer is to the question (0–1, higher is better)

---

## 2026-05-01T00:00:00Z — Baseline (pre-upgrade, 10 hardcoded questions)
- faithfulness:     0.8210
- answer_relevancy: 0.7380
- questions:        10
- note: Pre-upgrade baseline. Single ChromaDB MMR retriever (k=8, fetch_k=24). Regex guardrails only. In-memory session history. Scores from original project README.

## 2026-05-01T00:00:00Z — Post-upgrade (50 questions, hybrid search + cross-encoder + all upgrades)
- faithfulness:     (run `python -m src.monitoring.ragas_eval --record "Post all-upgrades" to populate`)
- answer_relevancy: (run eval to populate)
- questions:        50
- note: After Upgrades 1–5: BM25+vector EnsembleRetriever (0.6/0.4 weights), cross-encoder/ms-marco-MiniLM-L-6-v2 re-ranker, Llama Guard 3 second-stage (disabled by default), Redis session memory + JWT denylist, 50-question eval set (factual/multi_hop/ambiguous/no_answer/edge_case).

## 2026-05-02T20:11:47Z — Post all-upgrades (25q subset): hybrid BM25+vector,            
  cross-encoder re-ranker
- faithfulness:     nan
- answer_relevancy: 0.7381
- questions:        25
- note: Post all-upgrades (25q subset): hybrid BM25+vector,            
  cross-encoder re-ranker
