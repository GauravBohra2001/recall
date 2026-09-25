"""LangGraph wiring: one graph, three real human gates.

    router -> memory_read -+-> [conflict gate] -> retrieve -> assess <-> secondary_retrieve
                           |                                    |
                           |                                    v
                           |                     draft (validate + retry) -> [brief approval] -> memory_write
                           |
                           +-> (claims query) claims_retrieve -> claims_draft -> [claim selection] -> memory_write

Every bracketed node calls langgraph.types.interrupt(). The graph is checkpointed to
SQLite, so it stops there and cannot continue until the caller resumes the same
thread_id with Command(resume=...). memory_write is downstream of every gate, so
nothing reaches client_memory without a human action.

Rule for gate nodes: LangGraph re-runs a node from its first line when it resumes, so
nothing before interrupt() may have side effects. Logging happens after it.
"""

import sqlite3
from copy import deepcopy
from functools import partial

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agents.claims_agent import claims_draft_node, claims_retrieve_node
from agents.memory_agent import (
    _decision_line,
    memory_read_node,
    memory_write_node,
)
from agents.state import RecallState
from agents.strategy_agent import (
    assess_node,
    draft_node,
    retrieve_node,
    secondary_retrieve_node,
)
from db.database import DB_PATH, log_action
from rag.router import is_claims_query, route_query

UNANSWERABLE_MESSAGE = (
    "This question asks Winston to predict something HAZRA cannot observe. "
    "Recall only answers from detected signal, so no brief was produced. "
    "Try asking what the current signal supports instead."
)
CONFLICT_CANCELLED_MESSAGE = (
    "Session stopped at the conflict gate. The analyst chose not to proceed against a "
    "previously rejected direction, so no brief was drafted."
)


# ------------------------------------------------------------------ plain nodes
def router_node(state, conn):
    query_type = route_query(state["query"])
    log_action(
        conn,
        trace_id=state["trace_id"], client_id=state["client_id"],
        session_id=state["session_id"], agent_name="router",
        action_type="route_query", input_summary=state["query"][:200],
        retrieval_sources=[], retrieval_confidence=None, output_summary=query_type,
    )
    return {
        "query_type": query_type,
        "agent_log": [{
            "agent": "router", "action": "route_query",
            "ui_state": {"status": "memory_agent_running",
                         "client_id": state["client_id"], "query_type": query_type},
        }],
    }


def unanswerable_node(state, conn):
    log_action(
        conn,
        trace_id=state["trace_id"], client_id=state["client_id"],
        session_id=state["session_id"], agent_name="router",
        action_type="refused_unanswerable", input_summary=state["query"][:200],
        retrieval_sources=[], retrieval_confidence=0.0,
        output_summary="unanswerable query, no retrieval attempted",
    )
    return {
        "error_message": UNANSWERABLE_MESSAGE,
        "agent_log": [{
            "agent": "router", "action": "refused_unanswerable",
            "ui_state": {"status": "unanswerable", "output": None},
        }],
    }


# ------------------------------------------------------------------ human gates
def human_conflict_gate_node(state, conn):
    """Gate 1. The memory agent found a past REJECTED direction close to this query.
    A human decides whether to proceed before the strategy agent does any work."""
    conflicts = [
        m for m in state["memory_context"]
        if m["decision_type"] in ("claim_rejected", "brief_rejected")
    ]
    decision = interrupt({
        "type": "conflict_gate",
        "finding": state["memory_finding"],
        "conflicts": [_decision_line(m) for m in conflicts],
        "context": [_decision_line(m) for m in state["memory_context"]],
    })

    proceed = decision.get("decision") == "proceed"
    log_action(
        conn,
        trace_id=state["trace_id"], client_id=state["client_id"],
        session_id=state["session_id"], agent_name="human_gate",
        action_type="conflict_gate_decision", input_summary=state["query"][:200],
        retrieval_sources=[m["decision_id"] for m in conflicts],
        retrieval_confidence=None,
        output_summary="analyst proceeds with conflict acknowledged" if proceed
        else "analyst stopped the session",
        human_decision="proceed" if proceed else "cancel",
    )
    return {
        "conflict_acknowledged": proceed,
        "error_message": None if proceed else CONFLICT_CANCELLED_MESSAGE,
        "agent_log": [{
            "agent": "human_gate", "action": "conflict_gate_decision",
            "ui_state": {"status": "conflict_acknowledged" if proceed
                         else "conflict_cancelled"},
        }],
    }


def human_brief_approval_node(state, conn):
    """Gate 2. Nothing proceeds past this point without an explicit approve or reject."""
    decision = interrupt({
        "type": "brief_approval",
        "brief": state["draft_brief"],
        "confidence": state["brief_confidence"],
        "citations": state["brief_citations"],
        "analyst_review_required": state["analyst_review_required"],
    })

    verdict = "approved" if decision.get("decision") == "approved" else "rejected"
    reason = (decision.get("reason") or "").strip() or None
    if verdict == "rejected" and not reason:
        reason = "No reason given."
    log_action(
        conn,
        trace_id=state["trace_id"], client_id=state["client_id"],
        session_id=state["session_id"], agent_name="human_gate",
        action_type="brief_decision", input_summary=state["query"][:200],
        retrieval_sources=state["brief_citations"],
        retrieval_confidence=state["brief_confidence"],
        output_summary=f"brief {verdict}", human_decision=verdict,
        rejection_reason=reason if verdict == "rejected" else None,
        analyst_review_required=state["analyst_review_required"],
    )
    return {
        "human_brief_decision": verdict,
        "brief_rejection_reason": reason if verdict == "rejected" else None,
        "agent_log": [{
            "agent": "human_gate", "action": "brief_decision",
            "ui_state": {"status": "human_decision_recorded", "human_decision": verdict},
        }],
    }


def human_claim_selection_node(state, conn):
    """Gate 3. The human sees flagged claims in red with the legal reason attached,
    rejects the ones that breach past decisions and selects one safe claim."""
    decision = interrupt({
        "type": "claim_selection",
        "claims": state["draft_claims"],
        "flagged": [c["claim_id"] for c in state["flagged_claims"]],
        "confidence": state["retrieval_confidence"],
    })

    claims = deepcopy(state["draft_claims"])
    by_id = {c["claim_id"]: c for c in claims}
    rejected_reasons = []
    for item in decision.get("rejected", []):
        claim = by_id.get(item.get("claim_id"))
        if claim:
            claim["rejected"] = True
            claim["rejection_reason"] = (
                (item.get("reason") or "").strip() or claim.get("flag_reason")
                or "Rejected by reviewer."
            )
            rejected_reasons.append(f"{claim['claim_id']}: {claim['rejection_reason']}")

    chosen = by_id.get(decision.get("selected_claim_id"))
    if chosen and (chosen.get("flagged") or chosen.get("rejected")):
        chosen = None  # a flagged or rejected claim can never be the selection

    for claim in claims:
        if claim.get("rejected"):
            log_action(
                conn,
                trace_id=state["trace_id"], client_id=state["client_id"],
                session_id=state["session_id"], agent_name="human_gate",
                action_type="claim_decision", input_summary=state["query"][:200],
                retrieval_sources=[claim.get("matched_decision_id")],
                retrieval_confidence=None, output_summary=f"{claim['claim_id']} rejected",
                human_decision="claim_rejected", rejection_reason=claim["rejection_reason"],
            )
    log_action(
        conn,
        trace_id=state["trace_id"], client_id=state["client_id"],
        session_id=state["session_id"], agent_name="human_gate",
        action_type="claim_decision", input_summary=state["query"][:200],
        retrieval_sources=[], retrieval_confidence=None,
        output_summary=f"selected {chosen['claim_id']}" if chosen else "no valid selection",
        human_decision="claim_selected" if chosen else "no_selection",
    )
    return {
        "draft_claims": claims,
        "selected_claim": chosen["claim_id"] if chosen else None,
        "claim_rejection_reason": "; ".join(rejected_reasons) or None,
        "agent_log": [{
            "agent": "human_gate", "action": "claim_decision",
            "ui_state": {"status": "human_decision_recorded",
                         "selected": chosen["claim_id"] if chosen else None,
                         "rejected": [c["claim_id"] for c in claims if c.get("rejected")]},
        }],
    }


# ---------------------------------------------------------------------- routing
def _after_router(state):
    return "unanswerable" if state["query_type"] == "unanswerable" else "memory_read"


def _after_memory(state):
    if is_claims_query(state["query"]):
        return "claims_retrieve"
    if state["memory_finding"] == "conflict_found":
        return "human_conflict_gate"
    return "retrieve"


def _after_conflict_gate(state):
    return "retrieve" if state["conflict_acknowledged"] else END


def _after_assess(state):
    return "secondary_retrieve" if state["secondary_retrieval_needed"] else "draft"


def _after_draft(state):
    return "human_brief_approval" if state.get("draft_brief") else END


def _after_claims_draft(state):
    return "human_claim_selection" if state.get("draft_claims") else END


def build_graph(conn, checkpoint_conn=None):
    """Compile the graph. `conn` is the app connection; the checkpointer gets its own
    connection to the same recall.db so approval state survives a server restart."""
    checkpoint_conn = checkpoint_conn or sqlite3.connect(
        DB_PATH, check_same_thread=False, timeout=30
    )
    checkpointer = SqliteSaver(checkpoint_conn)

    g = StateGraph(RecallState)
    for name, fn in [
        ("router", router_node),
        ("unanswerable", unanswerable_node),
        ("memory_read", memory_read_node),
        ("human_conflict_gate", human_conflict_gate_node),
        ("retrieve", retrieve_node),
        ("assess", assess_node),
        ("secondary_retrieve", secondary_retrieve_node),
        ("draft", draft_node),
        ("human_brief_approval", human_brief_approval_node),
        ("claims_retrieve", claims_retrieve_node),
        ("claims_draft", claims_draft_node),
        ("human_claim_selection", human_claim_selection_node),
        ("memory_write", memory_write_node),
    ]:
        g.add_node(name, partial(fn, conn=conn))

    g.set_entry_point("router")
    g.add_conditional_edges("router", _after_router,
                            {"unanswerable": "unanswerable", "memory_read": "memory_read"})
    g.add_edge("unanswerable", END)
    g.add_conditional_edges("memory_read", _after_memory, {
        "claims_retrieve": "claims_retrieve",
        "human_conflict_gate": "human_conflict_gate",
        "retrieve": "retrieve",
    })
    g.add_conditional_edges("human_conflict_gate", _after_conflict_gate,
                            {"retrieve": "retrieve", END: END})
    g.add_edge("retrieve", "assess")
    g.add_conditional_edges("assess", _after_assess, {
        "secondary_retrieve": "secondary_retrieve", "draft": "draft"})
    g.add_edge("secondary_retrieve", "assess")
    g.add_conditional_edges("draft", _after_draft,
                            {"human_brief_approval": "human_brief_approval", END: END})
    g.add_edge("human_brief_approval", "memory_write")
    g.add_edge("claims_retrieve", "claims_draft")
    g.add_conditional_edges("claims_draft", _after_claims_draft, {
        "human_claim_selection": "human_claim_selection", END: END})
    g.add_edge("human_claim_selection", "memory_write")
    g.add_edge("memory_write", END)

    return g.compile(checkpointer=checkpointer)
