"""Query classification. Pure Python, no model call."""

from typing import List


def route_query(query: str) -> str:
    query_lower = query.lower()

    unanswerable_patterns = [
        "competitor launch", "will competitor", "future launch",
        "when will", "predict", "forecast next quarter",
    ]
    if any(p in query_lower for p in unanswerable_patterns):
        return "unanswerable"

    memory_patterns = [
        "what did we decide", "past decision", "before", "previously",
        "last time", "have we tried", "did we approve",
    ]
    if any(p in query_lower for p in memory_patterns):
        return "memory"

    urgency_patterns = [
        "how long", "when should", "time to act", "window",
        "how much time", "urgency", "deadline",
    ]
    if any(p in query_lower for p in urgency_patterns):
        return "urgency"

    action_patterns = [
        "what should we do", "recommend", "suggest", "strategy",
        "what direction", "how should", "whitespace", "opportunity",
        "draft", "marketing claim", "claim",
    ]
    if any(p in query_lower for p in action_patterns):
        return "synthesis"

    return "signal"


def is_claims_query(query: str) -> bool:
    """True when the query asks for marketing claims rather than a strategy brief."""
    q = query.lower()
    return ("claim" in q) or ("draft a marketing" in q) or ("copy" in q and "draft" in q)


def is_whitespace_query(query: str) -> bool:
    """Whitespace / competitive questions are the ones that justify a second retrieval pass."""
    q = query.lower()
    return any(t in q for t in ["whitespace", "white space", "competitive", "gap", "positioning"])


def source_types_present(signals: List[dict]) -> set:
    return {s.get("source_type") for s in signals}
