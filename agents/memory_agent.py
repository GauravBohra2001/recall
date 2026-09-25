"""Memory agent. Runs before the strategy agent on every session.

Reads past decisions for this client only, scores them against the current query with
sentence-transformer cosine similarity, and reports one of:

  conflict_found   the query resembles a direction the client previously REJECTED
  precedent_found  the query resembles a direction the client previously APPROVED
  no_history       nothing similar on file

conflict_found is routed to a human conflict gate (an interrupt) before any strategy
work happens. Run scripts/tune_thresholds.py to see the real similarity numbers.
"""

import time
from typing import List, Optional, Tuple

from db.database import log_action, write_memory
from rag.retrieval import memory_search
from rag.router import is_claims_query, is_whitespace_query

# Similarity thresholds are the PRD's own (conflict 0.80, precedent 0.60). scripts/tune_thresholds.py
# showed that lower values cannot work: Query 1 ("muscle health opportunity") scores 0.77 against
# the rejected muscle claim, higher than the whitespace question does, so a low conflict threshold
# would fire on the wrong demo query.
PRECEDENT_SIMILARITY_THRESHOLD = 0.60
CONFLICT_SIMILARITY_THRESHOLD = 0.80

# A whitespace or positioning question asks Winston to recommend a direction, and the open
# whitespace in HAZ-2024-GLP1-002 is a *clinical-language* claim, which is the direction legal
# already rejected. So for these questions any related rejection is a conflict, at a lower bar.
POSITIONING_CONFLICT_FLOOR = 0.35

APPROVAL_TYPES = {"brief_approved", "claim_selected"}
REJECTION_TYPES = {"brief_rejected", "claim_rejected"}


def classify_finding(memory_context: List[dict], query: str = "") -> Tuple[str, Optional[dict]]:
    """Returns (finding, the record that drove it)."""
    rejections = [m for m in memory_context if m["decision_type"] in REJECTION_TYPES]
    approvals = [m for m in memory_context if m["decision_type"] in APPROVAL_TYPES
                 and m["similarity_score"] >= PRECEDENT_SIMILARITY_THRESHOLD]

    close = [m for m in rejections if m["similarity_score"] >= CONFLICT_SIMILARITY_THRESHOLD]
    if close:
        return "conflict_found", max(close, key=lambda m: m["similarity_score"])

    if is_whitespace_query(query):
        related = [m for m in rejections if m["similarity_score"] >= POSITIONING_CONFLICT_FLOOR]
        if related:
            return "conflict_found", max(related, key=lambda m: m["similarity_score"])

    if approvals:
        # A strategy question's precedent is a past brief decision, a claims question's is a
        # past claim decision. Prefer the matching kind, then the closest.
        kind = "claim_" if is_claims_query(query) else "brief_"
        preferred = [m for m in approvals if m["decision_type"].startswith(kind)] or approvals
        return "precedent_found", max(preferred, key=lambda m: m["similarity_score"])

    return "no_history", None


def _decision_line(m: dict) -> str:
    label = m["decision_type"].replace("_", " ")
    line = f"{m['decision_id']} ({m['created_at']}, {label}): {m['direction']}"
    if m.get("rejection_reason"):
        line += f" | Reason: {m['rejection_reason']}"
    return line


def memory_banner_text(state: dict) -> Optional[str]:
    finding = state.get("memory_finding")
    ctx = state.get("memory_context") or []
    if finding == "no_history" or not ctx:
        return None
    header = (
        "Memory conflict: this query resembles a direction previously rejected."
        if finding == "conflict_found"
        else "Past decision found. Memory context is incorporated in this brief."
    )
    relevant = [m for m in ctx
                if m["decision_type"] in APPROVAL_TYPES | REJECTION_TYPES]
    return header + "\n\n" + "\n".join(f"- {_decision_line(m)}" for m in relevant)


def memory_read_node(state, conn):
    start = time.time()
    ranked = memory_search(state["query"], state["client_id"], conn, top_k=50,
                           min_similarity=0.3)
    # Keep the three closest decisions, plus the two closest rejections even if approvals
    # would crowd them out: as approvals accumulate, a rejection must never silently
    # drop out of context, or a real conflict stops being detected.
    ctx = ranked[:3]
    for m in [r for r in ranked if r["decision_type"] in REJECTION_TYPES][:2]:
        if m not in ctx:
            ctx.append(m)
    finding, driver = classify_finding(ctx, state["query"])
    latency = int((time.time() - start) * 1000)

    log_action(
        conn,
        trace_id=state["trace_id"], client_id=state["client_id"],
        session_id=state["session_id"], agent_name="memory_agent",
        action_type="memory_read", input_summary=state["query"][:200],
        retrieval_sources=[m["decision_id"] for m in ctx],
        retrieval_confidence=max((m["similarity_score"] for m in ctx), default=0.0),
        output_summary=(f"{finding}" + (f" via {driver['decision_id']}" if driver else "")),
        latency_ms=latency,
    )

    return {
        "memory_context": ctx,
        "memory_finding": finding,
        "agent_log": [{
            "agent": "memory_agent", "action": "memory_read",
            "ui_state": {
                "status": "memory_finding",
                "finding": finding,
                "related_decision_id": driver["decision_id"] if driver else None,
                "similarity": driver["similarity_score"] if driver else None,
            },
        }],
    }


def memory_write_node(state, conn):
    """Persists every human decision. It sits downstream of every interrupt, so nothing
    reaches client_memory without an explicit human action having resumed the graph."""
    written = []

    def _write(decision_type, direction, reason=None, human_decision=None):
        decision_id = write_memory(
            conn, state["client_id"], state["session_id"], decision_type,
            state["query"][:300], direction, reason,
        )
        written.append(decision_id)
        log_action(
            conn,
            trace_id=state["trace_id"], client_id=state["client_id"],
            session_id=state["session_id"], agent_name="memory_agent",
            action_type="memory_write", input_summary=state["query"][:200],
            retrieval_sources=[decision_id], retrieval_confidence=None,
            output_summary=f"{decision_type}: {direction[:120]}",
            human_decision=human_decision or decision_type, rejection_reason=reason,
        )

    if state.get("draft_claims"):
        for claim in state["draft_claims"]:
            if claim.get("rejected"):
                _write("claim_rejected", claim["claim_text"],
                       claim.get("rejection_reason") or "Rejected by reviewer.")
        if state.get("selected_claim"):
            chosen = next((c for c in state["draft_claims"]
                           if c["claim_id"] == state["selected_claim"]), None)
            if chosen:
                _write("claim_selected", chosen["claim_text"])
    elif state.get("human_brief_decision") in ("approved", "rejected"):
        import json
        brief = json.loads(state["draft_brief"])
        direction = brief.get("recommended_direction", "")[:500]
        if state["human_brief_decision"] == "approved":
            _write("brief_approved", direction)
        else:
            _write("brief_rejected", direction, state.get("brief_rejection_reason"))

    return {
        "memory_written": bool(written),
        "agent_log": [{
            "agent": "memory_agent", "action": "memory_write",
            "ui_state": {"status": "complete", "memory_written": bool(written),
                         "decision_ids": written},
        }],
    }
