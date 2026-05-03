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
- note: Pre-upgrade baseline. Single ChromaDB MMR retriever (k=8, fetch_k=24). Regex guardrails only. In-memory session history.

## 2026-05-03T21:31:10Z — Post all-upgrades (25q subset): hybrid BM25+vector, cross-encoder re-ranker
- faithfulness:     0.9116
- answer_relevancy: 0.7346
- questions:        25
- note: After all upgrades: BM25+vector EnsembleRetriever, cross-encoder re-ranker, Redis session memory + JWT denylist, Groq llama-3.3-70b-versatile for RAG, llama-3.3-70b-versatile for RAGAS scoring.

## 2026-05-03T22:45:58Z — Post all-upgrades (50q full): hybrid BM25+vector, cross-encoder re-ranker
- faithfulness:     0.7423
- answer_relevancy: 0.6495
- questions:        50
- note: Full 50-question set. RAGAS scoring via llama-3.1-8b-instant (scores are a conservative lower bound — 8b produces noisier statement extraction than 70b).
