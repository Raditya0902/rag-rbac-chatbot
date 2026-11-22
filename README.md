# RAG RBAC Chatbot

An internal knowledge base chatbot with Retrieval-Augmented Generation (RAG) and Role-Based Access Control (RBAC). Employees can only query documents from their own department — an Engineering employee cannot access Finance data, a Marketing employee cannot access HR records, and so on.

---

## Features

- **Role-Based Access Control** — both halves of retrieval (BM25 index and ChromaDB query) are restricted to the user's departments, and chat history is private to each user; tested per role against the real documents
- **Hybrid Retrieval** — BM25 + vector (ChromaDB MMR) ensemble with cross-encoder re-ranking for higher precision
- **JWT Authentication** — secure token-based auth with expiry and server-side token revocation (Redis denylist)
- **Streaming Responses** — token-by-token streaming via Server-Sent Events (SSE)
- **Conversation Memory** — Redis-backed multi-turn chat with question condensing for follow-ups
- **Input Guardrails** — blocks prompt injection, jailbreaks, SQL injection, XSS, and off-topic queries
- **Output Guardrails** — PII redaction (SSN, email, phone, credit card) from all responses
- **Llama Prompt Guard 2** — optional second-stage injection classifier via Groq (disabled by default; blocks at `GUARD_THRESHOLD`, default 0.5)
- **User Feedback Loop** — 👍👎 buttons in UI wired to LangSmith for continuous evaluation
- **Rate Limiting** — per-user rate limits (not just per-IP) on all endpoints
- **RAGAS Evaluation** — 50 questions: faithfulness 0.9490 / answer relevancy 0.8831 on the 40 answerable ones, and 10/10 correct refusals (DeepSeek judge; see [caveats](#what-these-scores-do-and-dont-show))
- **Incremental Re-ingestion** — MD5-based change detection; only re-indexes modified documents
- **Structured Logging** — JSON logs with latency, session IDs, guardrail events
- **191 Automated Tests** — RBAC isolation, chat-history isolation, guardrails, re-ranking, Redis denylist, re-ingestion, eval scoring
- **Docker** — one-command deployment with health checks

---

## Architecture

```
User
 │
 ├─ POST /auth/login ────────────────────► JWT Token
 │
 └─ POST /chat/stream  (Bearer token)
         │
         ├─ Rate Limiter          (15 req/min per user)
         ├─ JWT Validation + Denylist check (Redis)
         ├─ Input Guardrail       (injection, off-topic, sanitize)
         ├─ Question Condenser    (follow-ups → standalone question)
         ├─ RBAC Retriever        (BM25 + ChromaDB MMR ensemble, namespace filter)
         ├─ Cross-Encoder Re-ranker
         ├─ RAG Chain             (LLM + Redis conversation history)
         ├─ Output Guardrail      (PII redaction)
         └─ SSE Stream ─────────────────► Browser
                 │
                 └─ LangSmith Tracing + Structured Logs
```

### RBAC Namespace Map

| Role | Accessible Departments |
|------|----------------------|
| CLevel | Engineering, Finance, HR, Marketing, General |
| Engineering | Engineering, General |
| Finance | Finance, General |
| HR | HR, General |
| Marketing | Marketing, General |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | `openai/gpt-oss-120b` via Groq (Llama 3.3 70B until Groq retired it) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | ChromaDB (persistent, MMR search) |
| BM25 | rank-bm25 |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| RAG Framework | LangChain |
| Session Memory | Redis (2-hour TTL, 5-turn window) |
| API | FastAPI + uvicorn |
| Auth | JWT (python-jose, bcrypt) + Redis denylist |
| Frontend | Streamlit |
| Evaluation | RAGAS; judge LLM configurable (DeepSeek `deepseek-chat` or Groq Llama) |
| Observability | LangSmith + structlog |
| Rate Limiting | slowapi |
| Deployment | Docker + docker-compose |

---

## Project Structure

```
rag-rbac-chatbot/
├── src/
│   ├── api/
│   │   ├── auth.py              # POST /auth/login, /auth/logout
│   │   ├── chat.py              # POST /chat, /chat/stream, /feedback
│   │   └── limiter.py           # per-user rate limiter
│   ├── auth/
│   │   ├── denylist.py          # Redis-backed JWT revocation
│   │   ├── jwt.py               # token create/decode
│   │   ├── rbac.py              # role → namespace map
│   │   └── users.py             # user store + bcrypt auth
│   ├── guardrails/
│   │   ├── input_guard.py       # injection detection, domain filter, Prompt Guard 2
│   │   └── output_guard.py      # PII redaction
│   ├── ingestion/
│   │   ├── ingest.py            # full ingest — populates ChromaDB
│   │   ├── loaders.py           # markdown + CSV loaders with MD5 hashing
│   │   └── splitter.py          # recursive character splitter
│   ├── monitoring/
│   │   ├── answer_progress.py   # saves eval answers as they arrive so runs can resume
│   │   ├── judge.py             # RAGAS judge LLM (Groq or DeepSeek)
│   │   ├── langsmith_setup.py   # feedback logging to LangSmith
│   │   ├── ragas_eval.py        # RAGAS faithfulness + answer relevancy
│   │   ├── refusal.py           # refusal accuracy for no_answer questions
│   │   └── score_report.py      # per-question scores by type and role
│   ├── rag/
│   │   ├── chain.py             # run_query + stream_query
│   │   └── prompts.py           # RAG + condense prompts
│   ├── vectorstore/
│   │   └── store.py             # hybrid retriever + cross-encoder re-ranker
│   ├── config.py
│   └── logging_config.py
├── frontend/
│   └── streamlit_app.py
├── tests/
│   ├── test_rbac_isolation.py   # 60 tests — no role retrieves another department's chunks
│   ├── test_session_isolation.py # 5 tests — chat history is private to its user
│   ├── test_guardrails.py       # 56 tests — injection, PII, domain filter
│   ├── test_retrieval.py        # 7 tests — cross-encoder re-ranking
│   ├── test_redis.py            # 8 tests — JWT denylist
│   ├── test_reingest.py         # 8 tests — incremental re-ingestion
│   └── test_ragas_eval.py, …    # 47 tests — eval pipeline, judge, refusal, score reports, resume
├── resources/
│   └── data/                    # mock company documents
│       ├── engineering/
│       ├── finance/
│       ├── general/
│       ├── hr/
│       └── marketing/
├── reingest.py                  # incremental re-ingestion (changed files only)
├── main.py
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Setup

### Prerequisites
- Python 3.11+
- Redis running locally (or via Docker)
- A [Groq API key](https://console.groq.com) (free tier) for the RAG chain
- For evaluation, a judge key: Groq (free) or [DeepSeek](https://platform.deepseek.com) (paid, a few cents per 50-question run); see [Evaluation](#evaluation)
- Optionally a [LangSmith API key](https://smith.langchain.com) for tracing

### 1. Clone and install

```bash
git clone https://github.com/Raditya0902/rag-rbac-chatbot.git
cd rag-rbac-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set GROQ_API_KEY and JWT_SECRET_KEY
```

### 3. Ingest documents into ChromaDB

```bash
python -m src.ingestion.ingest
```

To re-index only changed documents after edits:

```bash
python reingest.py
```

### 4. Run

**Terminal 1 — Backend:**
```bash
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```

Open `http://localhost:8501`

---

## Docker

```bash
cp .env.example .env   # fill in your API keys
docker compose up --build
```

Both services start automatically. The frontend waits for the backend health check before launching.

---

## Test Credentials

| Username | Password | Role |
|----------|----------|------|
| `ceo` | `ceosecret` | CLevel (all departments) |
| `tony` | `password123` | Engineering |
| `sam` | `financepass` | Finance |
| `natasha` | `hrpass123` | HR |
| `bruce` | `securepass` | Marketing |

---

## Evaluation

RAGAS scoring uses a "judge" LLM chosen by `EVAL_JUDGE_PROVIDER` and `EVAL_JUDGE_MODEL`:

| Provider | Default model | Key | Notes |
|----------|---------------|-----|-------|
| `groq` (default) | `openai/gpt-oss-20b` | `GROQ_API_KEY` | Free tier; shares Groq's rate limits with the RAG chain |
| `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` | Paid, but a full 50-question run costs cents and avoids Groq quota limits |

Override either per run with `--judge-provider` / `--judge-model`. The judge is recorded with each entry in `EVAL_RESULTS.md`. Only compare scores produced by the same judge on the same question set.

**How questions are scored.** `no_answer` questions (facts the documents don't contain) are scored by **refusal accuracy**: whether the bot replied with the prompt's exact refusal sentence. They're kept out of RAGAS because RAGAS gives a correct "I don't know" 0 on both metrics. Every other question is scored by RAGAS; an answerable question that gets refused counts as a **false refusal** and still scores 0 there. Each run also writes per-question scores, broken down by question type and role, to `eval_scores.json`.

```bash
# Run against the 25-question subset (recommended — fits within free-tier limits)
python -m src.monitoring.ragas_eval --subset --record "description of what changed"

# If the eval phase fails mid-run, retry scoring without re-running RAG queries
python -m src.monitoring.ragas_eval --eval-only --record "description of what changed"

# Full 50-question set
python -m src.monitoring.ragas_eval --record "description of what changed"

# Hit a rate limit or daily token cap partway? Answers are saved as they arrive
# (eval_samples_progress.json), so running the same command again resumes.
# Saved answers are discarded if the prompt or RAG model has changed since.

# Re-score cached answers with a larger Groq judge (shares the RAG chain's token quota)
python -m src.monitoring.ragas_eval --eval-only --judge-model openai/gpt-oss-120b --record "gpt-oss-120b judge"

# Re-score cached answers with DeepSeek
python -m src.monitoring.ragas_eval --eval-only --judge-provider deepseek --record "deepseek judge"
```

Results are appended to `EVAL_RESULTS.md`.

### Current scores (full 50-question set, hybrid BM25+vector + cross-encoder re-ranker)

| Metric | Score |
|--------|-------|
| Faithfulness (40 answerable) | **0.9490** |
| Answer Relevancy (40 answerable) | **0.8831** |
| Refusal accuracy (10 `no_answer`) | **10/10** |
| False refusals (40 answerable) | **0/40** |

Fresh answers from the current pipeline (`openai/gpt-oss-120b`, facts-first prompt), judged by `deepseek-chat`, every sample scored. The set includes deliberately hard `multi_hop`, `ambiguous` and `no_answer` questions.

| By type | n | Faithfulness | Relevancy |
|---------|---|--------------|-----------|
| factual | 15 | 0.9778 | 0.9543 |
| ambiguous | 10 | 0.9592 | 0.8564 |
| edge_case | 5 | 0.9167 | 0.9424 |
| multi_hop | 10 | 0.9117 | 0.7732 |

#### What these scores do and don't show

- **The facts-first prompt trades a little relevancy for faithfulness.** With the same model and judge, the earlier complete-sentence prompt scored 0.8916 / 0.9048 (multi-hop 0.7440 / 0.8298). gpt-oss-120b tended to state links between facts the documents list separately, so the prompt now forbids unstated purposes, causes and connections: the model gives the facts it has, then says "The documents don't describe how these are connected." RAGAS reads that sentence as noncommittal and lowers relevancy, which is the cost. A stricter variant that didn't insist on giving the facts first reached 0.9566 faithfulness but fell to 0.8556 relevancy, with two answers at 0 (one run, same model and judge, not recorded in `EVAL_RESULTS.md`).
- **Multi-hop answers are still the weak spot.** The model still sometimes states a link the documents don't: that check-ins are "intended to monitor" progress against probation objectives, or that the gateway's authorizers "interact with" the Authentication Service. Each number is also a single run with an LLM judge; one fully correct factual answer scored 0.67 faithfulness this run.
- **The model change wasn't measured on its own.** Groq retired Llama 3.3 70B, so the model moved to gpt-oss-120b in the same run as the complete-sentence prompt. The previous Llama answers (from 2026-05-03, same judge) scored 0.9639 / 0.8572, so relevancy rose and faithfulness fell, but the run can't separate the model's effect from the prompt's.
- **Scoring refusals separately raises the averages.** RAGAS gives a correct "I don't know" 0, so `no_answer` questions are scored by refusal instead. With all 50 through RAGAS, the 2026-05-03 answers scored 0.7810 / 0.6931 rather than 0.9639 / 0.8572. Earlier entries in `EVAL_RESULTS.md` also used other judges or question sets (the 25-question subset scored 0.9116 / 0.7346), so they aren't directly comparable.
- **Refusal accuracy is an exact-sentence match.** 10/10 means the bot gave the prescribed refusal on those questions. It doesn't measure anything finer.
- **The question set is small and self-written, and the judge is an LLM.** Treat the numbers as a regression check for this project, not a benchmark.
- **RBAC isn't measured by RAGAS.** It's covered by `tests/test_rbac_isolation.py` and `tests/test_session_isolation.py`.

---

## Tests

```bash
pytest tests/ -v
```

191 tests covering:
- RBAC isolation: for every role, the BM25 index and the ChromaDB query hold only that role's departments, checked with questions aimed at other departments (the ChromaDB tests skip if the store hasn't been ingested)
- Chat-history isolation: two users sending the same session id get separate histories
- Injection detection, off-topic filtering, follow-up handling, HTML sanitization, PII redaction
- Cross-encoder re-ranking order and metadata preservation
- Redis JWT denylist revocation and failure handling
- Incremental re-ingestion change detection and MD5 hashing
- Eval pipeline: judge selection, refusal scoring, per-question reports, unscored-sample handling, rate-limit waits and resuming an interrupted run

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/login` | None | Returns JWT token |
| POST | `/auth/logout` | Bearer | Revokes token server-side |
| POST | `/chat` | Bearer | Standard chat |
| POST | `/chat/stream` | Bearer | Streaming chat (SSE) |
| POST | `/feedback` | Bearer | Submit 👍👎 feedback |
| GET | `/health` | None | Health check |
