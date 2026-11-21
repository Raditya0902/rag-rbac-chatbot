# RAG RBAC Chatbot

An enterprise-grade internal knowledge base chatbot built with Retrieval-Augmented Generation (RAG) and Role-Based Access Control (RBAC). Employees can only query documents from their own department — an Engineering employee cannot access Finance data, a Marketing employee cannot access HR records, and so on.

---

## Features

- **Role-Based Access Control** — retrieval is filtered at the vector store level per user role, not just the UI
- **Hybrid Retrieval** — BM25 + vector (ChromaDB MMR) ensemble with cross-encoder re-ranking for higher precision
- **JWT Authentication** — secure token-based auth with expiry and server-side token revocation (Redis denylist)
- **Streaming Responses** — token-by-token streaming via Server-Sent Events (SSE)
- **Conversation Memory** — Redis-backed multi-turn chat with question condensing for follow-ups
- **Input Guardrails** — blocks prompt injection, jailbreaks, SQL injection, XSS, and off-topic queries
- **Output Guardrails** — PII redaction (SSN, email, phone, credit card) from all responses
- **Llama Guard 3** — optional second-stage AI safety check via Groq (disabled by default)
- **User Feedback Loop** — 👍👎 buttons in UI wired to LangSmith for continuous evaluation
- **Rate Limiting** — per-user rate limits (not just per-IP) on all endpoints
- **RAGAS Evaluation** — faithfulness + answer relevancy scoring on 50 questions: faithfulness 0.7810 / answer relevancy 0.6931 (DeepSeek judge, all samples scored)
- **Incremental Re-ingestion** — MD5-based change detection; only re-indexes modified documents
- **Structured Logging** — JSON logs with latency, session IDs, guardrail events
- **77 Automated Tests** — guardrails, retrieval re-ranking, Redis denylist, re-ingestion logic
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
| LLM | Llama 3.3 70B via Groq |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | ChromaDB (persistent, MMR search) |
| BM25 | rank-bm25 |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| RAG Framework | LangChain |
| Session Memory | Redis (2-hour TTL, 5-turn window) |
| API | FastAPI + uvicorn |
| Auth | JWT (python-jose, bcrypt) + Redis denylist |
| Frontend | Streamlit |
| Evaluation | RAGAS + Groq llama-3.1-8b-instant |
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
│   │   ├── input_guard.py       # injection detection, domain filter, Llama Guard 3
│   │   └── output_guard.py      # PII redaction
│   ├── ingestion/
│   │   ├── ingest.py            # full ingest — populates ChromaDB
│   │   ├── loaders.py           # markdown + CSV loaders with MD5 hashing
│   │   └── splitter.py          # recursive character splitter
│   ├── monitoring/
│   │   ├── langsmith_setup.py   # feedback logging to LangSmith
│   │   └── ragas_eval.py        # RAGAS faithfulness + answer relevancy
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
│   ├── test_guardrails.py       # 44 tests — injection, PII, domain filter
│   ├── test_retrieval.py        # 9 tests — cross-encoder re-ranking
│   ├── test_redis.py            # 11 tests — JWT denylist
│   └── test_reingest.py         # 13 tests — incremental re-ingestion
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
- A [Groq API key](https://console.groq.com) (free tier — used for both the RAG chain and RAGAS evaluation)
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
| `groq` (default) | `llama-3.1-8b-instant` | `GROQ_API_KEY` | Free tier; 8b has its own daily token pool, separate from the 70b RAG chain |
| `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` | Paid, but a full 50-question run costs cents and avoids Groq quota limits |

Override either per run with `--judge-provider` / `--judge-model`. The judge is recorded with each entry in `EVAL_RESULTS.md`. Only compare scores produced by the same judge on the same question set.

```bash
# Run against the 25-question subset (recommended — fits within free-tier limits)
python -m src.monitoring.ragas_eval --subset --record "description of what changed"

# If the eval phase fails mid-run, retry scoring without re-running RAG queries
python -m src.monitoring.ragas_eval --eval-only --record "description of what changed"

# Full 50-question set
python -m src.monitoring.ragas_eval --record "description of what changed"

# Re-score cached answers with the 70b judge (uses the RAG chain's token quota)
python -m src.monitoring.ragas_eval --eval-only --judge-model llama-3.3-70b-versatile --record "70b judge"

# Re-score cached answers with DeepSeek
python -m src.monitoring.ragas_eval --eval-only --judge-provider deepseek --record "deepseek judge"
```

Results are appended to `EVAL_RESULTS.md`.

### Current scores (full 50-question set, hybrid BM25+vector + cross-encoder re-ranker)

| Metric | Score |
|--------|-------|
| Faithfulness | **0.7810** |
| Answer Relevancy | **0.6931** |

Judge: `deepseek-chat`, 50/50 samples scored. The set includes deliberately hard `multi_hop`, `ambiguous` and `no_answer` questions. Earlier entries in `EVAL_RESULTS.md` used other judges or question sets (the 25-question subset scored 0.9116 / 0.7346), so they aren't directly comparable.

---

## Tests

```bash
pytest tests/ -v
```

77 tests covering:
- Injection detection, off-topic filtering, follow-up handling, HTML sanitization, PII redaction
- Cross-encoder re-ranking order and metadata preservation
- Redis JWT denylist revocation and failure handling
- Incremental re-ingestion change detection and MD5 hashing

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
