"""Strategy agent. Retrieves, decides whether it has enough signal, loops if not, then
drafts a cited brief and validates it before any human sees it.

The loop is what makes this an agent rather than a pipeline: after the first retrieval
it inspects what it got, names the gap, formulates a targeted second query, retrieves
again and reassesses. The trigger is a gap in source coverage (a whitespace question
answered without any consumer-language signal), not a confidence threshold. Confidence
is a weighted average of signals that already scored 0.79 to 0.94, so a threshold
would almost never fire; a coverage gap is deterministic and explainable.
"""

import json
import time

from agents.llm import call_llm_with_retry, model_label
from config.prompts import BRIEF_SCHEMA, SYSTEM_PROMPT, build_brief_user_message
from data.hazra_signals import HAZRA_SIGNALS
from db.database import log_action
from rag.retrieval import compute_confidence, hybrid_search, structured_timing_lookup
from rag.router import is_whitespace_query
from rag.validation import (
    CONFIDENCE_FLOOR,
    extract_citations,
    retry_feedback,
    validate_citations,
)

PRIMARY_TOP_K = 3
MAX_RETRIEVAL_ATTEMPTS = 2
MAX_VALIDATION_ATTEMPTS = 3

GAP_SOURCE_TYPE = "consumer_language_analysis"
GAP_DESCRIPTION = "no consumer language signal retrieved for a whitespace question"
SECONDARY_QUERY = (
    "consumer language preference maintenance preservation framing muscle GLP-1"
)


def _ids(signals):
    return [s["signal_id"] for s in signals]


def _merge(existing, new_signals):
    seen = {s["signal_id"] for s in existing}
    return existing + [s for s in new_signals if s["signal_id"] not in seen]


def retrieve_node(state, conn):
    start = time.time()
    if state["query_type"] == "urgency":
        signals = structured_timing_lookup(HAZRA_SIGNALS)
        method = "structured_lookup"
    else:
        signals = hybrid_search(state["query"], HAZRA_SIGNALS, top_k=PRIMARY_TOP_K)
        method = "hybrid_rrf"
    confidence = compute_confidence(signals)

    log_action(
        conn,
        trace_id=state["trace_id"], client_id=state["client_id"],
        session_id=state["session_id"], agent_name="strategy_agent",
        action_type="retrieval_attempt_1", input_summary=state["query"][:200],
        retrieval_sources=_ids(signals), retrieval_confidence=confidence,
        output_summary=f"{len(signals)} signals via {method}",
        latency_ms=int((time.time() - start) * 1000),
    )
    return {
        "hazra_signals": signals,
        "retrieval_confidence": confidence,
        "retrieval_attempts": 1,
        "agent_log": [{
            "agent": "strategy_agent", "action": "retrieval_attempt_1",
            "ui_state": {
                "status": "strategy_agent_retrieving", "attempt": 1,
                "signals": _ids(signals), "retrieval_confidence": confidence,
            },
        }],
    }


def assess_node(state, conn):
    """The agentic decision: is this enough signal to draft on, or is there a gap?"""
    signals = state["hazra_signals"]
    have = {s.get("source_type") for s in signals}
    gap = None
    if (state["retrieval_attempts"] < MAX_RETRIEVAL_ATTEMPTS
            and is_whitespace_query(state["query"])
            and GAP_SOURCE_TYPE not in have):
        gap = GAP_DESCRIPTION

    needed = gap is not None
    log_action(
        conn,
        trace_id=state["trace_id"], client_id=state["client_id"],
        session_id=state["session_id"], agent_name="strategy_agent",
        action_type="sufficiency_check", input_summary=state["query"][:200],
        retrieval_sources=_ids(signals), retrieval_confidence=state["retrieval_confidence"],
        output_summary=f"secondary_retrieval_needed={needed}" + (f"; gap={gap}" if gap else ""),
    )
    return {
        "secondary_retrieval_needed": needed,
        "secondary_retrieval_query": SECONDARY_QUERY if needed else "",
        "agent_log": [{
            "agent": "strategy_agent", "action": "sufficiency_check", "gap": gap,
            "ui_state": {
                "status": "sufficiency_check",
                "secondary_retrieval_needed": needed,
                "gap_identified": gap,
            },
        }],
    }


def secondary_retrieve_node(state, conn):
    """Targeted second pass: search only the source type the first pass was missing."""
    start = time.time()
    have = _ids(state["hazra_signals"])
    pool = [s for s in HAZRA_SIGNALS
            if s["source_type"] == GAP_SOURCE_TYPE and s["signal_id"] not in have]
    extra = hybrid_search(state["secondary_retrieval_query"], pool, top_k=2)
    signals = _merge(state["hazra_signals"], extra)
    confidence = compute_confidence(signals)

    log_action(
        conn,
        trace_id=state["trace_id"], client_id=state["client_id"],
        session_id=state["session_id"], agent_name="strategy_agent",
        action_type="retrieval_attempt_2", input_summary=state["secondary_retrieval_query"],
        retrieval_sources=_ids(signals), retrieval_confidence=confidence,
        output_summary=f"added {_ids(extra)} after gap: {GAP_DESCRIPTION}",
        latency_ms=int((time.time() - start) * 1000),
    )
    log_action(
        conn,
        trace_id=state["trace_id"], client_id=state["client_id"],
        session_id=state["session_id"], agent_name="strategy_agent",
        action_type="secondary_retrieval", input_summary=GAP_DESCRIPTION,
        retrieval_sources=_ids(extra), retrieval_confidence=confidence,
        output_summary=f"secondary pass retrieved {_ids(extra)}; confidence now {confidence}",
    )
    return {
        "hazra_signals": signals,
        "retrieval_confidence": confidence,
        "retrieval_attempts": state["retrieval_attempts"] + 1,
        "secondary_retrieval_needed": False,
        "agent_log": [{
            "agent": "strategy_agent", "action": "retrieval_attempt_2",
            "ui_state": {
                "status": "secondary_retrieval", "attempt": 2,
                "gap_identified": GAP_DESCRIPTION,
                "signals_added": _ids(extra),
                "new_retrieval_confidence": confidence,
            },
        }],
    }


def _failure(state, conn, action_type, message, summary, confidence, latency=0, mode=None):
    log_action(
        conn,
        trace_id=state["trace_id"], client_id=state["client_id"],
        session_id=state["session_id"], agent_name="strategy_agent",
        action_type=action_type, input_summary=state["query"][:200],
        retrieval_sources=_ids(state["hazra_signals"]), retrieval_confidence=confidence,
        output_summary=summary[:300], latency_ms=latency,
        model_used=model_label(mode) if mode else None,
    )
    return {
        "draft_brief": None,
        "error_message": message,
        "agent_log": [{"agent": "strategy_agent", "action": action_type,
                       "ui_state": {"status": "no_output", "error": message}}],
    }


def draft_node(state, conn):
    """Draft, validate, and retry with the named errors. The analyst only ever sees a
    brief that passed validation, or an honest error."""
    signals = state["hazra_signals"]
    confidence = state["retrieval_confidence"]

    if confidence < CONFIDENCE_FLOOR:
        return _failure(
            state, conn, "refused_low_confidence",
            f"Retrieval confidence is {confidence:.2f}, below the {CONFIDENCE_FLOOR:.2f} "
            "floor. Winston will not draft a brief on this little signal. Try a more "
            "specific question about the detected GLP-1 signals.",
            f"confidence {confidence} below floor", confidence,
        )

    user_message = build_brief_user_message(
        state["query"], signals, state["memory_context"], confidence
    )
    if state.get("conflict_acknowledged"):
        user_message += (
            "\n\nThe analyst reviewed the conflict between this query and a previously "
            "rejected direction and chose to proceed. Name the conflicting decision id "
            "explicitly in memory_incorporated and explain how the brief avoids it."
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    valid_ids = _ids(signals)
    errors, mode = [], None

    for attempt in range(1, MAX_VALIDATION_ATTEMPTS + 1):
        result = call_llm_with_retry(messages, BRIEF_SCHEMA)
        mode = result.mode
        if not result.ok:
            return _failure(
                state, conn, "draft_failed",
                "Winston could not reach a working model after three attempts, so no "
                "brief was produced. Nothing was shown ungrounded. Please try again.",
                f"LLM failure: {result.error}", confidence, mode=mode,
            )

        brief = result.parsed
        brief["confidence_score"] = confidence  # the computed value, not the model's guess
        log_action(
            conn,
            trace_id=state["trace_id"], client_id=state["client_id"],
            session_id=state["session_id"], agent_name="strategy_agent",
            action_type=f"llm_draft_attempt_{attempt}", input_summary=state["query"][:200],
            retrieval_sources=extract_citations(brief), retrieval_confidence=confidence,
            output_summary=brief.get("executive_summary", "")[:300],
            latency_ms=result.latency_ms, model_used=model_label(mode),
            token_count=result.token_count,
        )

        check = validate_citations(brief, valid_ids, confidence)
        log_action(
            conn,
            trace_id=state["trace_id"], client_id=state["client_id"],
            session_id=state["session_id"], agent_name="validator",
            action_type="citation_validation_passed" if check["valid"]
            else "citation_validation_failed",
            input_summary=f"draft attempt {attempt}", retrieval_sources=valid_ids,
            retrieval_confidence=confidence,
            output_summary=("passed" if check["valid"] else "; ".join(check["errors"]))[:300]
            + f" | coverage={check['citation_coverage_rate']:.2f}",
            analyst_review_required=check["analyst_review_required"],
        )

        if check["valid"]:
            citations = extract_citations(brief)
            return {
                "draft_brief": json.dumps(brief),
                "brief_citations": citations,
                "brief_confidence": confidence,
                "citation_validation_passed": True,
                "analyst_review_required": check["analyst_review_required"],
                "agent_log": [
                    {"agent": "strategy_agent", "action": "brief_drafted",
                     "ui_state": {
                         "status": "brief_drafted",
                         "retrieval_confidence": confidence,
                         "citations": citations,
                         "analyst_review_required": check["analyst_review_required"],
                         "validation_attempts": attempt,
                     }},
                    {"agent": "human_gate", "action": "awaiting_human_approval",
                     "ui_state": {"status": "awaiting_human_approval",
                                  "human_decision": None}},
                ],
            }

        errors = check["errors"]
        messages = messages + [
            {"role": "assistant", "content": json.dumps(result.parsed)},
            {"role": "user", "content": retry_feedback(errors, valid_ids)},
        ]

    return _failure(
        state, conn, "validation_exhausted",
        "Winston could not produce a fully cited brief after three attempts, so nothing "
        "was shown. Validation errors on the last attempt: " + "; ".join(errors),
        "; ".join(errors), confidence, mode=mode,
    )
