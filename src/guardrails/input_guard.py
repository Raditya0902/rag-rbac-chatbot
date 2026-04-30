import html
import re

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
    "employee", "employees", "engineering", "finance", "hr", "human", "resources",
    "marketing", "salary", "salaries", "policy", "policies", "project", "budget",
    "report", "department", "role", "team", "hire", "hiring", "performance",
    "benefits", "onboarding", "training", "revenue", "expense", "quarter",
    "annual", "company", "internal", "handbook", "code", "software", "data",
    "leave", "vacation", "compliance", "security", "product", "customer",
}


def sanitize(query: str) -> str:
    query = html.unescape(query)
    query = re.sub(r"<[^>]+>", " ", query)
    return re.sub(r"\s+", " ", query).strip()


def check_input(query: str, is_followup: bool = False) -> tuple[bool, str]:
    """Returns (is_ok, block_reason). is_ok=False means the query should be blocked.

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
    return True, ""
