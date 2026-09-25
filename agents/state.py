"""The typed state object every LangGraph node reads from and writes to."""

import operator
from typing import Annotated, List, Optional, TypedDict


class RecallState(TypedDict):
    client_id: str
    session_id: str
    query: str
    query_type: str
    hazra_signals: List[dict]
    memory_context: List[dict]
    memory_finding: str
    conflict_acknowledged: bool
    retrieval_confidence: float
    retrieval_attempts: int
    secondary_retrieval_needed: bool
    secondary_retrieval_query: str
    draft_brief: Optional[str]
    brief_citations: List[str]
    citation_validation_passed: bool
    brief_confidence: float
    analyst_review_required: bool
    human_brief_decision: Optional[str]
    brief_rejection_reason: Optional[str]
    draft_claims: Optional[List[dict]]
    claims_direction_concern: bool
    flagged_claims: List[dict]
    selected_claim: Optional[str]
    claim_rejection_reason: Optional[str]
    memory_written: bool
    trace_id: str
    error_message: Optional[str]
    # Reducer: nodes return only the entries they add and LangGraph appends them.
    agent_log: Annotated[List[dict], operator.add]


def new_state(client_id: str, session_id: str, trace_id: str, query: str) -> RecallState:
    return {
        "client_id": client_id,
        "session_id": session_id,
        "trace_id": trace_id,
        "query": query,
        "query_type": "",
        "hazra_signals": [],
        "memory_context": [],
        "memory_finding": "no_history",
        "conflict_acknowledged": False,
        "retrieval_confidence": 0.0,
        "retrieval_attempts": 0,
        "secondary_retrieval_needed": False,
        "secondary_retrieval_query": "",
        "draft_brief": None,
        "brief_citations": [],
        "citation_validation_passed": False,
        "brief_confidence": 0.0,
        "analyst_review_required": False,
        "human_brief_decision": None,
        "brief_rejection_reason": None,
        "draft_claims": None,
        "claims_direction_concern": False,
        "flagged_claims": [],
        "selected_claim": None,
        "claim_rejection_reason": None,
        "memory_written": False,
        "error_message": None,
        "agent_log": [],
    }
