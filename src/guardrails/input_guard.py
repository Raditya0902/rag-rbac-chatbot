import html
import re

import structlog
from groq import Groq

from src.config import settings

log = structlog.get_logger()

INJECTION_PATTERNS = [
    # Broad form catches typos: "ignore * instructions/prompts"
    re.compile(r"ignore\s+\S+(\s+\S+)?\s+(instructions?|prompts?)", re.I),
    re.compile(r"\breveal\s+(all\s+)?(data|information|secrets?|credentials?)\b", re.I),
    re.compile(r"<script", re.I),
    re.compile(r"(drop|delete|truncate)\s+table", re.I),
    re.compile(r"union\s+select", re.I),
    re.compile(r"system\s*prompt", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a)\s+", re.I),
    re.compile(r"(you\s+are\s+now|from\s+now\s+on)\s+", re.I),
]

DOMAIN_KEYWORDS = {
    # org / people
    "employee", "employees", "department", "role", "team", "hire", "hiring",
    "performance", "onboarding", "training", "handbook", "company", "internal",
    "values", "core", "mission", "vision", "business", "ceo", "director",
    # HR / leave / benefits
    "hr", "human", "resources", "salary", "salaries", "benefits", "leave",
    "vacation", "sick", "maternity", "paternity", "bereavement", "statutory",
    "bonus", "gratuity", "provident", "insurance", "compensation",
    # finance
    "finance", "revenue", "expense", "budget", "quarter", "annual", "report",
    "income", "profit", "margin", "dividends", "shareholders", "cash", "gross",
    "operating", "fiscal", "cost", "costs", "pressures", "price", "financial",
    # engineering / tech
    "engineering", "software", "code", "data", "security", "compliance",
    "cloud", "infrastructure", "architecture", "database", "programming",
    "mobile", "applications", "authentication", "api", "deployment",
    # marketing
    "marketing", "product", "customer", "campaign", "acquisition", "retention",
    # work arrangements
    "home", "remote", "policy", "policies", "project",
}

# Llama Prompt Guard 2 takes the raw text to classify — no special template needed.
# It returns "INJECTION" for prompt injection attempts, "BENIGN" otherwise.


def sanitize(query: str) -> str:
    query = html.unescape(query)
    query = re.sub(r"<[^>]+>", " ", query)
    return re.sub(r"\s+", " ", query).strip()


def _llama_guard_check(query: str) -> bool:
    """Return True if Llama Prompt Guard 2 flags the query as an injection attempt.

    Passes the raw query text directly — no special prompt template.
    The model responds with 'INJECTION' or 'BENIGN'.
    Fails open (returns False = safe) on any API error so a Groq outage
    cannot be used as a DoS vector against the service itself.
    """
    try:
        client = Groq(api_key=settings.GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=settings.GUARD_MODEL,
            messages=[{"role": "user", "content": query}],
            max_tokens=10,
            temperature=0,
        )
        verdict = resp.choices[0].message.content.strip().upper()
        return verdict.startswith("INJECTION")
    except Exception as exc:
        log.warning("llama_guard_error", error=str(exc))
        return False  # fail open


def check_input(query: str, is_followup: bool = False) -> tuple[bool, str]:
    """Returns (is_ok, block_reason). is_ok=False means the query should be blocked.

    Pipeline:
      1. Length check
      2. Regex injection patterns (always runs)
      3. Domain keyword check (skipped for follow-up turns)
      4. Llama Prompt Guard 2 via Groq (runs only when ENABLE_GUARD_MODEL=true)

    is_followup=True skips the domain keyword check — follow-up messages like
    "summarize that" are valid in context but contain no company keywords.
    Injection checks always run regardless.
    """
    if not (3 <= len(query) <= 2000):
        return False, "Query must be between 3 and 2000 characters."

    for pattern in INJECTION_PATTERNS:
        if pattern.search(query):
            return False, "Your query contains disallowed patterns."

    if not is_followup:
        tokens = set(re.findall(r"\b\w+\b", query.lower()))
        if not tokens & DOMAIN_KEYWORDS:
            return False, "Your query appears to be off-topic. Please ask about company-related information."

    if settings.ENABLE_GUARD_MODEL and _llama_guard_check(query):
        log.warning("llama_guard_blocked", query_preview=query[:80])
        return False, "Your query contains disallowed patterns."

    return True, ""
