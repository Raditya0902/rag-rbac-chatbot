import time
import uuid
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path

import structlog
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.config import settings
from src.guardrails.input_guard import check_input, sanitize
from src.guardrails.output_guard import redact_pii
from src.rag.prompts import CONDENSE_PROMPT, RAG_PROMPT
from src.vectorstore.store import get_rbac_retriever

# In-memory history store keyed by session_id.
# Each value is a list of (human_text, ai_text) turn tuples.
# Note: single-process only — use Redis or a DB for multi-worker deployments.
_histories: dict[str, list[tuple[str, str]]] = defaultdict(list)
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


def _to_messages(turns: list[tuple[str, str]]) -> list:
    msgs = []
    for human, ai in turns:
        msgs.append(HumanMessage(content=human))
        msgs.append(AIMessage(content=ai))
    return msgs


def _condense(question: str, history: list[tuple[str, str]]) -> str:
    """Rewrite a follow-up question into a standalone question for retrieval."""
    if not history:
        return question
    condensed = (CONDENSE_PROMPT | _get_llm() | StrOutputParser()).invoke({
        "history": _to_messages(history),
        "question": question,
    })
    return condensed.strip()


def run_query(query: str, allowed_namespaces: list[str], session_id: str | None = None) -> dict:
    session_id = session_id or str(uuid.uuid4())
    query_preview = query[:80]

    history = _histories[session_id]
    is_ok, reason = check_input(query, is_followup=bool(history))
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
    retrieval_query = _condense(sanitized, history)

    retriever = get_rbac_retriever(allowed_namespaces)
    docs = retriever.invoke(retrieval_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    run_id = uuid.uuid4()
    t0 = time.perf_counter()

    answer_raw = (RAG_PROMPT | _get_llm() | StrOutputParser()).invoke(
        {"context": context, "history": _to_messages(history), "input": sanitized},
        config={"run_id": run_id, "run_name": f"rag_{session_id}"},
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    answer = redact_pii(answer_raw)

    history.append((sanitized, answer))
    _histories[session_id] = history[-_HISTORY_WINDOW:]

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

    history = _histories[session_id]
    is_ok, reason = check_input(query, is_followup=bool(history))
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
    retrieval_query = _condense(sanitized, history)

    retriever = get_rbac_retriever(allowed_namespaces)
    docs = retriever.invoke(retrieval_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    run_id = uuid.uuid4()
    yield f"__run_id__:{run_id}"

    t0 = time.perf_counter()
    full_answer = ""
    for chunk in (RAG_PROMPT | _get_llm() | StrOutputParser()).stream(
        {"context": context, "history": _to_messages(history), "input": sanitized},
        config={"run_id": run_id, "run_name": f"rag_{session_id}"},
    ):
        full_answer += chunk
        yield chunk

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    history.append((sanitized, redact_pii(full_answer)))
    _histories[session_id] = history[-_HISTORY_WINDOW:]

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
