"""Claims agent. Drafts three options and flags any that resemble a past rejection,
so the human sees the flag and the legal reason before choosing.

A claim is flagged when ANY check fires:
  - string similarity to a past rejected claim (difflib) is 0.70 or higher
  - it contains a phrase HAZ-2024-GLP1-005 names as a regulatory risk
  - it is the aggressive option and the client has a past rejected claim on file. The
    drafting prompt makes option 1 the strongest efficacy wording, and a real model
    routinely phrases it without the exact rejected words ("helps rebuild muscle"), so
    exact-phrase checks alone miss it. Efficacy-style language is what legal rejected.

Embedding similarity is deliberately NOT a flag trigger. Measured on this data it scores
a safe claim ("Supports muscle maintenance on GLP-1 medication") at 0.91 against the
rejected "Clinically proven to build muscle" claim, the same as a risky paraphrase at
0.91: embeddings capture topic, not hedging, so they would flag the claim the analyst
should be selecting. Embeddings are used only to attribute a flag to the right past
rejection.
"""

import json
import time
from difflib import SequenceMatcher

from sklearn.metrics.pairwise import cosine_similarity

from agents.llm import call_llm_with_retry, model_label
from config.prompts import CLAIMS_SCHEMA, CLAIMS_SYSTEM_PROMPT, build_claims_user_message
from data.hazra_signals import HAZRA_SIGNALS
from db.database import log_action
from rag.retrieval import compute_confidence, embed, hybrid_search
from rag.validation import CITATION_RE, CONFIDENCE_FLOOR

STRING_FLAG_THRESHOLD = 0.70
ATTRIBUTION_MIN_SIMILARITY = 0.45
MAX_DRAFT_ATTEMPTS = 3
REJECTION_TYPES = ("claim_rejected", "brief_rejected")

# Risky patterns from HAZ-2024-GLP1-005, plus the language legal already rejected in
# MEM-HAL-2024-002.
BANNED_PHRASES = ["clinically proven", "clinically shown", "scientifically proven",
                  "demonstrated to", "proven", "prevents", "guaranteed", "cures"]


def _ids(signals):
    return [s["signal_id"] for s in signals]


def claims_retrieve_node(state, conn):
    start = time.time()
    signals = hybrid_search(state["query"], HAZRA_SIGNALS, top_k=3)
    confidence = compute_confidence(signals)
    log_action(
        conn,
        trace_id=state["trace_id"], client_id=state["client_id"],
        session_id=state["session_id"], agent_name="claims_agent",
        action_type="retrieval_attempt_1", input_summary=state["query"][:200],
        retrieval_sources=_ids(signals), retrieval_confidence=confidence,
        output_summary=f"{len(signals)} signals via hybrid_rrf",
        latency_ms=int((time.time() - start) * 1000),
    )
    return {
        "hazra_signals": signals,
        "retrieval_confidence": confidence,
        "retrieval_attempts": 1,
        "agent_log": [{
            "agent": "claims_agent", "action": "retrieval_attempt_1",
            "ui_state": {"status": "claims_agent_running", "attempt": 1,
                         "signals": _ids(signals), "retrieval_confidence": confidence},
        }],
    }


def _past_rejections(conn, client_id):
    rows = conn.execute(
        "SELECT * FROM client_memory WHERE client_id = ? AND decision_type IN (?, ?)",
        (client_id, *REJECTION_TYPES),
    ).fetchall()
    return [dict(r) for r in rows]


def flag_claims(claims, rejections):
    """Marks each claim in place. Returns (claims, flagged_subset)."""
    flagged = []
    sims = None
    if claims and rejections:
        sims = cosine_similarity(
            embed([c["claim_text"] for c in claims]),
            embed([r["direction"] for r in rejections]),
        )

    for i, claim in enumerate(claims):
        text = claim["claim_text"].lower()
        claim.update(flagged=False, flag_reason=None, matched_decision_id=None,
                     match_similarity=0.0, rejected=False)

        best_j, best_score, best_string = None, 0.0, 0.0
        for j, rej in enumerate(rejections):
            emb = float(sims[i][j]) if sims is not None else 0.0
            string = SequenceMatcher(None, text, rej["direction"].lower()).ratio()
            score = max(emb, string)
            if score > best_score:
                best_j, best_score, best_string = j, score, string

        phrase = next((p for p in BANNED_PHRASES if p in text), None)
        similar = best_j is not None and best_string >= STRING_FLAG_THRESHOLD
        efficacy = claim.get("claim_strength") == "aggressive" and bool(rejections)
        if not (phrase or similar or efficacy):
            continue

        match = None
        if best_j is not None and (phrase or similar):
            rej = rejections[best_j]
            rej_text = (rej["direction"] + " " + (rej["rejection_reason"] or "")).lower()
            if best_score >= ATTRIBUTION_MIN_SIMILARITY or (phrase and phrase in rej_text):
                match = rej

        claim["flagged"] = True
        claim["match_similarity"] = round(best_score, 3)
        if match:
            claim["matched_decision_id"] = match["decision_id"]
            claim["flag_reason"] = (
                f"Matches past rejection {match['decision_id']} ({match['created_at']}). "
                f"{match.get('rejection_reason') or ''}".strip()
            )
        elif phrase:
            claim["flag_reason"] = (
                f"Contains risky language '{phrase}', which HAZ-2024-GLP1-005 identifies "
                "as a high regulatory rejection pattern."
            )
        else:
            # efficacy rule: attribute to the closest past rejection, else the newest
            past = rejections[best_j] if best_j is not None else rejections[-1]
            claim["matched_decision_id"] = past["decision_id"]
            claim["flag_reason"] = (
                "This claim uses efficacy-style language similar to the previously "
                "rejected clinically proven direction. Legal rejected this language in "
                f"{past['decision_id']}."
            )
        flagged.append(claim)

    return claims, flagged


def _claim_problems(parsed, valid_ids):
    claims = parsed.get("claims") or []
    problems = []
    if len(claims) != 3:
        problems.append(f"WRONG_COUNT: expected exactly 3 claims, got {len(claims)}")
    for claim in claims:
        tags = CITATION_RE.findall(claim.get("rationale", ""))
        if not tags:
            problems.append(f"UNCITED_RATIONALE: {claim.get('claim_id')}")
        for tag in tags:
            if tag.strip() not in valid_ids:
                problems.append(f"GHOST_CITATION: {tag.strip()} not in retrieved signals")
    return problems


def claims_draft_node(state, conn):
    signals = state["hazra_signals"]
    confidence = state["retrieval_confidence"]
    valid_ids = _ids(signals)

    def fail(action_type, message, summary, mode=None):
        log_action(
            conn,
            trace_id=state["trace_id"], client_id=state["client_id"],
            session_id=state["session_id"], agent_name="claims_agent",
            action_type=action_type, input_summary=state["query"][:200],
            retrieval_sources=valid_ids, retrieval_confidence=confidence,
            output_summary=summary[:300], model_used=model_label(mode) if mode else None,
        )
        return {"draft_claims": None, "error_message": message,
                "agent_log": [{"agent": "claims_agent", "action": action_type,
                               "ui_state": {"status": "no_output", "error": message}}]}

    if confidence < CONFIDENCE_FLOOR:
        return fail(
            "refused_low_confidence",
            f"Retrieval confidence is {confidence:.2f}, below the {CONFIDENCE_FLOOR:.2f} "
            "floor. Winston will not draft claims on this little signal.",
            f"confidence {confidence} below floor",
        )

    messages = [
        {"role": "system", "content": CLAIMS_SYSTEM_PROMPT},
        {"role": "user", "content": build_claims_user_message(
            state["query"], signals, state["memory_context"])},
    ]
    problems, mode, result = [], None, None

    for attempt in range(1, MAX_DRAFT_ATTEMPTS + 1):
        result = call_llm_with_retry(messages, CLAIMS_SCHEMA)
        mode = result.mode
        if not result.ok:
            return fail(
                "draft_failed",
                "Winston could not reach a working model after three attempts, so no "
                "claims were drafted. Please try again.",
                f"LLM failure: {result.error}", mode,
            )
        log_action(
            conn,
            trace_id=state["trace_id"], client_id=state["client_id"],
            session_id=state["session_id"], agent_name="claims_agent",
            action_type=f"llm_draft_attempt_{attempt}", input_summary=state["query"][:200],
            retrieval_sources=valid_ids, retrieval_confidence=confidence,
            output_summary=f"{len(result.parsed.get('claims', []))} claims returned",
            latency_ms=result.latency_ms, model_used=model_label(mode),
            token_count=result.token_count,
        )
        problems = _claim_problems(result.parsed, set(valid_ids))
        log_action(
            conn,
            trace_id=state["trace_id"], client_id=state["client_id"],
            session_id=state["session_id"], agent_name="validator",
            action_type="citation_validation_failed" if problems
            else "citation_validation_passed",
            input_summary=f"claims draft attempt {attempt}", retrieval_sources=valid_ids,
            retrieval_confidence=confidence,
            output_summary=("; ".join(problems) if problems else "passed")[:300],
        )
        if not problems:
            break
        messages = messages + [
            {"role": "assistant", "content": json.dumps(result.parsed)},
            {"role": "user", "content": (
                "Your previous output failed validation. Fix exactly these problems:\n"
                + "\n".join(f"- {p}" for p in problems)
                + f"\nReturn exactly three claims. Valid signal_ids: {', '.join(valid_ids)}."
            )},
        ]
    else:
        return fail(
            "validation_exhausted",
            "Winston could not produce three fully cited claims after three attempts. "
            "Problems on the last attempt: " + "; ".join(problems),
            "; ".join(problems), mode,
        )

    parsed = result.parsed
    rejections = _past_rejections(conn, state["client_id"])
    claims, flagged = flag_claims(parsed["claims"], rejections)

    log_action(
        conn,
        trace_id=state["trace_id"], client_id=state["client_id"],
        session_id=state["session_id"], agent_name="claims_agent",
        action_type="claims_flagged_against_memory", input_summary=state["query"][:200],
        retrieval_sources=[c["matched_decision_id"] for c in flagged if c["matched_decision_id"]],
        retrieval_confidence=confidence,
        output_summary=f"{len(claims)} claims drafted, {len(flagged)} flagged: "
        + ", ".join(c["claim_id"] for c in flagged),
        model_used=model_label(mode),
    )
    return {
        "draft_claims": claims,
        "flagged_claims": flagged,
        "claims_direction_concern": bool(parsed.get("direction_concern")) or bool(flagged),
        "agent_log": [{
            "agent": "claims_agent", "action": "claims_drafted",
            "ui_state": {
                "status": "awaiting_human_claim_selection",
                "claims": len(claims),
                "flagged": [c["claim_id"] for c in flagged],
            },
        }],
    }
