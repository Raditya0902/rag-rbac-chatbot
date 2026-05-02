import time
import uuid
from collections.abc import Generator
from pathlib import Path

import structlog
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.config import settings
from src.guardrails.input_guard import check_input, sanitize
from src.guardrails.output_guard import redact_pii
from src.rag.prompts import CONDENSE_PROMPT, RAG_PROMPT
from src.vectorstore.store import get_rbac_retriever, rerank_docs

# Session TTL: keep conversation history for 2 hours (in seconds).
_SESSION_TTL = 7200
# Keep only the last 5 turns (10 messages) in context to bound prompt size.
_HISTORY_WINDOW = 5

log = structlog.get_logger()

_NS_LABEL = {
    "engineering": "Engineering",
    "finance":     "Finance",
    "hr":          "HR",
    "marketing":   "Marketing",
    "general":     "General",
}


def _format_source(namespace: str, filename: str) -> str:
    """Turn ('finance', 'quarterly_financial_report.md') → 'Finance › Quarterly Financial Report'."""
    ns = _NS_LABEL.get(namespace, namespace.title())
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").title()
    return f"{ns} › {stem}"


def _get_llm() -> ChatGroq:
    return ChatGroq(model=settings.LLM_MODEL, api_key=settings.GROQ_API_KEY)


def _get_history(session_id: str) -> RedisChatMessageHistory:
    """Return a Redis-backed message history for this session.

    Falls back gracefully — if Redis is unreachable the RedisChatMessageHistory
    constructor raises; callers catch this and continue with an empty history.
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=_SESSION_TTL,
    )


def _load_messages(session_id: str) -> list:
    """Load the last _HISTORY_WINDOW turns from Redis, or [] on failure."""
    try:
        history = _get_history(session_id)
        msgs = history.messages
        # Each turn = 1 HumanMessage + 1 AIMessage; window covers last N turns.
        return msgs[-(  _HISTORY_WINDOW * 2):]
    except Exception as exc:
        log.warning("history_load_failed", session_id=session_id, error=str(exc))
        return []


def _save_turn(session_id: str, human: str, ai: str) -> None:
    """Append a turn to Redis history; silently skip on failure."""
    try:
        history = _get_history(session_id)
        history.add_messages([HumanMessage(content=human), AIMessage(content=ai)])
    except Exception as exc:
        log.warning("history_save_failed", session_id=session_id, error=str(exc))


def _condense(question: str, history_messages: list) -> str:
    """Rewrite a follow-up question into a standalone question for retrieval."""
    if not history_messages:
        return question
    condensed = (CONDENSE_PROMPT | _get_llm() | StrOutputParser()).invoke({
        "history": history_messages,
        "question": question,
    })
    return condensed.strip()


def run_query(query: str, allowed_namespaces: list[str], session_id: str | None = None, llm=None) -> dict:
    session_id = session_id or str(uuid.uuid4())
    query_preview = query[:80]

    history_messages = _load_messages(session_id)
    is_ok, reason = check_input(query, is_followup=bool(history_messages))
    if not is_ok:
        log.warning(
            "guardrail_blocked",
            session_id=session_id,
            reason=reason,
            query_preview=query_preview,
        )
        return {"answer": reason, "sources": [], "run_id": None,
                "guardrail_blocked": True, "block_reason": reason}

    sanitized = sanitize(query)
    retrieval_query = _condense(sanitized, history_messages)

    retriever = get_rbac_retriever(allowed_namespaces)
    candidates = retriever.invoke(retrieval_query)
    docs = rerank_docs(retrieval_query, candidates, settings.RETRIEVAL_TOP_K)
    context = "\n\n".join(doc.page_content for doc in docs)

    run_id = uuid.uuid4()
    t0 = time.perf_counter()

    answer_raw = (RAG_PROMPT | (llm or _get_llm()) | StrOutputParser()).invoke(
        {"context": context, "history": history_messages, "input": sanitized},
        config={"run_id": run_id, "run_name": f"rag_{session_id}"},
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    answer = redact_pii(answer_raw)

    _save_turn(session_id, sanitized, answer)

    sources = list({
        _format_source(doc.metadata.get("namespace", ""), doc.metadata.get("source", ""))
        for doc in docs
        if doc.metadata.get("source")
    })
    log.info(
        "query_complete",
        session_id=session_id,
        run_id=str(run_id),
        namespaces=allowed_namespaces,
        docs_retrieved=len(docs),
        latency_ms=latency_ms,
        query_preview=query_preview,
    )
    return {"answer": answer, "sources": sources, "run_id": str(run_id),
            "guardrail_blocked": False, "block_reason": ""}


def stream_query(
    query: str, allowed_namespaces: list[str], session_id: str | None = None
) -> Generator[str, None, None]:
    """Yield answer tokens one at a time.

    Protocol (each line is one SSE data payload):
      __run_id__:<uuid>       — first event, for feedback association
      <token>                 — answer text chunks
      __sources__:<s1>|<s2>  — source file paths
      [DONE]                  — terminal event (sent by the API layer)
    """
    session_id = session_id or str(uuid.uuid4())
    query_preview = query[:80]

    history_messages = _load_messages(session_id)
    is_ok, reason = check_input(query, is_followup=bool(history_messages))
    if not is_ok:
        log.warning(
            "guardrail_blocked",
            session_id=session_id,
            reason=reason,
            query_preview=query_preview,
        )
        yield reason
        return

    sanitized = sanitize(query)
    retrieval_query = _condense(sanitized, history_messages)

    retriever = get_rbac_retriever(allowed_namespaces)
    candidates = retriever.invoke(retrieval_query)
    docs = rerank_docs(retrieval_query, candidates, settings.RETRIEVAL_TOP_K)
    context = "\n\n".join(doc.page_content for doc in docs)

    run_id = uuid.uuid4()
    yield f"__run_id__:{run_id}"

    t0 = time.perf_counter()
    full_answer = ""
    for chunk in (RAG_PROMPT | _get_llm() | StrOutputParser()).stream(
        {"context": context, "history": history_messages, "input": sanitized},
        config={"run_id": run_id, "run_name": f"rag_{session_id}"},
    ):
        full_answer += chunk
        yield chunk

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    _save_turn(session_id, sanitized, redact_pii(full_answer))

    sources = "|".join({
        _format_source(doc.metadata.get("namespace", ""), doc.metadata.get("source", ""))
        for doc in docs
        if doc.metadata.get("source")
    })
    yield f"__sources__:{sources}"

    log.info(
        "stream_complete",
        session_id=session_id,
        run_id=str(run_id),
        namespaces=allowed_namespaces,
        docs_retrieved=len(docs),
        latency_ms=latency_ms,
        query_preview=query_preview,
    )
