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

## 2026-11-21T09:43:02Z — 50q full, re-scored with deepseek-chat judge (cached answers)
- faithfulness:     0.7810
- answer_relevancy: 0.6931
- questions:        50
- judge_provider:   deepseek
- judge_model:      deepseek-chat
- note: 50q full, re-scored with deepseek-chat judge (cached answers)

## 2026-11-21T09:56:36Z — 50q full, deepseek-chat judge, no_answer scored by refusal accuracy
- faithfulness:     0.9639
- answer_relevancy: 0.8572
- questions:        50 (40 answerable scored by RAGAS, 10 no_answer)
- refusal_accuracy: 10/10
- false_refusals:   0/40
- judge_provider:   deepseek
- judge_model:      deepseek-chat
- note: 50q full, deepseek-chat judge, no_answer scored by refusal accuracy
- answers: cached from the 2026-05-03 50q run (pipeline before the 2026-11-21 complete-sentence prompt change); same answers as the entry above, only the scoring differs

## 2026-11-21T20:15:56Z — 50q full re-run: gpt-oss-120b RAG model, complete-sentence prompt, deepseek-chat judge
- faithfulness:     0.8916
- answer_relevancy: 0.9048
- questions:        50 (40 answerable scored by RAGAS, 10 no_answer)
- refusal_accuracy: 10/10
- false_refusals:   0/40
- judge_provider:   deepseek
- judge_model:      deepseek-chat
- note: 50q full re-run: gpt-oss-120b RAG model, complete-sentence prompt, deepseek-chat judge
