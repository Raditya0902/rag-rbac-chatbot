# RAG RBAC Chatbot

An enterprise-grade internal knowledge base chatbot built with Retrieval-Augmented Generation (RAG) and Role-Based Access Control (RBAC). Employees can only query documents from their own department — an Engineering employee cannot access Finance data, a Marketing employee cannot access HR records, and so on.

---

## Features

- **Role-Based Access Control** — retrieval is filtered at the vector store level per user role, not just the UI
- **JWT Authentication** — secure token-based auth with expiry; no credentials sent on every request
- **Streaming Responses** — token-by-token streaming via Server-Sent Events (SSE)
- **Conversation Memory** — multi-turn chat with question condensing for follow-ups ("tell me more about that" works correctly)
- **Input Guardrails** — blocks prompt injection, jailbreaks, SQL injection, off-topic queries
- **Output Guardrails** — PII redaction (SSN, email, phone, credit card) from all responses
- **User Feedback Loop** — 👍👎 buttons in UI wired to LangSmith for continuous evaluation
- **Rate Limiting** — per-user rate limits (not just per-IP) on all endpoints
- **RAGAS Evaluation** — faithfulness 0.875, answer relevancy 0.738
- **Structured Logging** — JSON logs with latency, session IDs, guardrail events
- **44 Automated Tests** — full coverage of security-critical guardrail logic
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
         ├─ Rate Limiter        (15 req/min per user)
         ├─ JWT Validation
         ├─ Input Guardrail     (injection, off-topic, sanitize)
         ├─ Question Condenser  (follow-ups → standalone question)
         ├─ RBAC Retriever      (ChromaDB MMR, namespace filter)
         ├─ RAG Chain           (LLM + conversation history)
         ├─ Output Guardrail    (PII redaction)
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
| RAG Framework | LangChain |
| API | FastAPI + uvicorn |
| Auth | JWT (python-jose, bcrypt) |
| Frontend | Streamlit |
| Evaluation | RAGAS |
| Observability | LangSmith + structlog |
| Rate Limiting | slowapi |
| Deployment | Docker + docker-compose |

---

## Project Structure

```
rag-rbac-chatbot/
├── src/
│   ├── api/
│   │   ├── auth.py          # POST /auth/login
│   │   ├── chat.py          # POST /chat, /chat/stream, /feedback
│   │   └── limiter.py       # per-user rate limiter
│   ├── auth/
│   │   ├── jwt.py           # token create/decode
│   │   ├── rbac.py          # role → namespace map
│   │   └── users.py         # user store + bcrypt auth
│   ├── guardrails/
│   │   ├── input_guard.py   # injection detection, domain filter
│   │   └── output_guard.py  # PII redaction
│   ├── ingestion/
│   │   ├── ingest.py        # run once to populate ChromaDB
│   │   ├── loaders.py       # markdown + CSV loaders
│   │   └── splitter.py      # recursive character splitter
│   ├── monitoring/
│   │   ├── langsmith_setup.py
│   │   └── ragas_eval.py    # RAGAS faithfulness + answer relevancy
│   ├── rag/
│   │   ├── chain.py         # run_query + stream_query
│   │   └── prompts.py       # RAG + condense prompts
│   ├── vectorstore/
│   │   └── store.py         # ChromaDB + RBAC retriever
│   ├── config.py
│   └── logging_config.py
├── frontend/
│   └── streamlit_app.py
├── tests/
│   └── test_guardrails.py   # 44 tests
├── resources/
│   └── data/                # mock company documents
│       ├── engineering/
│       ├── finance/
│       ├── general/
│       ├── hr/
│       └── marketing/
├── main.py
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Setup

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier works)
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
# Edit .env and add your GROQ_API_KEY and JWT_SECRET_KEY
```

### 3. Ingest documents into ChromaDB

```bash
python -m src.ingestion.ingest
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

Run RAGAS evaluation (requires the backend to be running):

```bash
python -m src.monitoring.ragas_eval
```

| Metric | Score |
|--------|-------|
| Faithfulness | **0.875** |
| Answer Relevancy | **0.738** |

---

## Tests

```bash
pytest tests/ -v
```

44 tests covering injection detection, off-topic filtering, follow-up handling, HTML sanitization, and PII redaction.

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/login` | None | Returns JWT token |
| POST | `/chat` | Bearer | Standard chat |
| POST | `/chat/stream` | Bearer | Streaming chat (SSE) |
| POST | `/feedback` | Bearer | Submit 👍👎 feedback |
| GET | `/health` | None | Health check |
