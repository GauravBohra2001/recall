"""Citation and confidence checks. Pure Python, no model call.

Three checks, in the order the PRD gives them:
  1. every factual claim carries a [SOURCE: signal_id] tag (or says "insufficient signal data")
  2. every SOURCE tag points at a signal actually retrieved this session (ghost check)
  3. confidence floor: below 0.50 there is no output, only an honest error
"""

import re
from typing import List

CITATION_RE = re.compile(r"\[SOURCE: ([^\]]+)\]")

CONFIDENCE_FLOOR = 0.50
ANALYST_REVIEW_BELOW = 0.65
INSUFFICIENT = "insufficient signal data"


def validate_citations(draft_output: dict, retrieved_signal_ids: List[str],
                       confidence: float = None) -> dict:
    valid_ids = set(retrieved_signal_ids)
    errors = []

    claims = list(draft_output.get("key_findings", []) or [])
    if draft_output.get("recommended_direction"):
        claims.append(draft_output["recommended_direction"])

    if not draft_output.get("key_findings"):
        errors.append("NO_FINDINGS: key_findings is empty")

    for text in claims:
        tags = CITATION_RE.findall(text)
        if not tags and INSUFFICIENT not in text.lower():
            errors.append(f"UNCITED_CLAIM: {text[:100]}")
        for tag in tags:
            if tag.strip() not in valid_ids:
                errors.append(f"GHOST_CITATION: {tag.strip()} not in retrieved signals")

    # The summary is prose rather than a claim list, so it is not required to carry a
    # tag, but any tag it does carry must still be real.
    for tag in CITATION_RE.findall(draft_output.get("executive_summary", "") or ""):
        if tag.strip() not in valid_ids:
            errors.append(f"GHOST_CITATION: {tag.strip()} not in retrieved signals (summary)")

    if confidence is None:
        confidence = draft_output.get("confidence_score", 0) or 0
    if confidence < CONFIDENCE_FLOOR:
        errors.append(f"CONFIDENCE_TOO_LOW: {confidence} below minimum {CONFIDENCE_FLOOR}")

    uncited = len([e for e in errors if e.startswith("UNCITED")])
    return {
        "valid": not errors,
        "errors": errors,
        "analyst_review_required": confidence < ANALYST_REVIEW_BELOW,
        "citation_coverage_rate": 1.0 - (uncited / max(len(claims), 1)),
    }


def extract_citations(draft_output: dict) -> List[str]:
    """Every distinct signal_id cited anywhere in the brief, in order of appearance."""
    seen = []
    texts = list(draft_output.get("key_findings", []) or [])
    for field in ("recommended_direction", "executive_summary"):
        if draft_output.get(field):
            texts.append(draft_output[field])
    for text in texts:
        for cid in CITATION_RE.findall(text):
            cid = cid.strip()
            if cid not in seen:
                seen.append(cid)
    return seen


def retry_feedback(errors: List[str], valid_ids: List[str]) -> str:
    """The specific errors, named, appended to the prompt on a validation retry."""
    lines = "\n".join(f"- {e}" for e in errors)
    return (
        "YOUR PREVIOUS OUTPUT FAILED CITATION VALIDATION. Fix exactly these problems:\n"
        f"{lines}\n\n"
        f"The only valid signal_ids are: {', '.join(valid_ids)}.\n"
        "Every key finding and the recommended direction must end with "
        "[SOURCE: signal_id] using one of those ids, or say \"insufficient signal data\"."
    )
