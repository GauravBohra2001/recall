"""All LLM prompts and structured-output schemas.

SYSTEM_PROMPT is hardcoded exactly as specified in the PRD and must never be edited.
"""

SYSTEM_PROMPT = """You are Winston, CREWASIS strategic analysis assistant helping Haleon brand team with GLP-1 vitamins and supplements strategy.

CRITICAL RULES NEVER VIOLATE THESE:
1. Every factual claim MUST be followed immediately by a citation: [SOURCE: signal_id]
2. If you cannot cite a specific signal_id write "insufficient signal data" instead
3. Never use knowledge from your training data
4. Only use information from HAZRA signals provided in context
5. If memory context shows a past decision reference it explicitly

OUTPUT FORMAT return a JSON object with exactly these fields:
executive_summary: 2-3 sentences summarizing key finding and recommended direction
key_findings: list of strings each ending with [SOURCE: HAZ-XXXX-XXX]
recommended_direction: one clear direction with citation [SOURCE: HAZ-XXXX-XXX]
urgency_tier: act_this_quarter OR act_this_year OR monitor
confidence_score: float between 0 and 1
memory_incorporated: description of how past decisions influenced this brief or null"""

CLAIMS_SYSTEM_PROMPT = """You are Winston, CREWASIS's claims drafting assistant helping Haleon's brand team with GLP-1 vitamins and supplements strategy.

CRITICAL RULES: NEVER VIOLATE THESE
1. Draft exactly three distinct marketing claim options.
2. Option 1 is the aggressive option. It must make a definitive efficacy assertion about muscle
   preservation for GLP-1 users, using at least one of these phrases: "clinically shown",
   "proven to", "demonstrated to" or "scientifically proven". This is the wording a brand team
   would ask for and legal will review, so do not soften it.
   Option 2 must use moderate language (for example "helps" or "supports"). Option 3 must use
   conservative, maintenance-led language (for example "designed to support").
3. Every claim must carry a rationale with a citation: [SOURCE: signal_id]
4. Only use information from the HAZRA signals provided in the context. Never use training data.
5. Do not self-censor option 1. Legal review and memory checking happen downstream.

OUTPUT FORMAT: Return a JSON object with exactly these fields:
{
  "claims": [
    {
      "claim_id": "CLAIM-1",
      "claim_text": "The marketing claim itself",
      "claim_strength": "aggressive OR moderate OR conservative",
      "rationale": "Why this claim follows from the signals, with [SOURCE: HAZ-XXXX-XXX]"
    }
  ],
  "direction_concern": "A short note if the requested direction conflicts with past decisions, or null"
}"""

BRIEF_SCHEMA = {
    "name": "recall_brief",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "executive_summary",
            "key_findings",
            "recommended_direction",
            "urgency_tier",
            "confidence_score",
            "memory_incorporated",
        ],
        "properties": {
            "executive_summary": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "string"}},
            "recommended_direction": {"type": "string"},
            "urgency_tier": {
                "type": "string",
                "enum": ["act_this_quarter", "act_this_year", "monitor"],
            },
            "confidence_score": {"type": "number"},
            "memory_incorporated": {"type": ["string", "null"]},
        },
    },
}

CLAIMS_SCHEMA = {
    "name": "recall_claims",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["claims", "direction_concern"],
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["claim_id", "claim_text", "claim_strength", "rationale"],
                    "properties": {
                        "claim_id": {"type": "string"},
                        "claim_text": {"type": "string"},
                        "claim_strength": {
                            "type": "string",
                            "enum": ["aggressive", "moderate", "conservative"],
                        },
                        "rationale": {"type": "string"},
                    },
                },
            },
            "direction_concern": {"type": ["string", "null"]},
        },
    },
}


def format_signals(signals) -> str:
    lines = []
    for s in signals:
        lines.append(
            f"- signal_id: {s['signal_id']}\n"
            f"  source_type: {s['source_type']}\n"
            f"  date_range: {s['date_range']}\n"
            f"  momentum: {s['hazra_momentum_score']} | confidence: {s['hazra_confidence']}\n"
            f"  text: {s['signal_text']}"
        )
    return "\n".join(lines) if lines else "No signals retrieved."


def format_memory(memory_context) -> str:
    if not memory_context:
        return "No relevant past decisions for this client."
    lines = []
    for m in memory_context:
        lines.append(
            f"- {m['decision_id']} ({m['created_at']}) type={m['decision_type']}\n"
            f"  direction: {m['direction']}\n"
            f"  rejection_reason: {m.get('rejection_reason') or 'none'}"
        )
    return "\n".join(lines)


def build_brief_user_message(query, signals, memory_context, retrieval_confidence) -> str:
    return (
        f"CLIENT QUERY:\n{query}\n\n"
        f"RETRIEVED HAZRA SIGNALS:\n{format_signals(signals)}\n\n"
        f"CLIENT MEMORY CONTEXT:\n{format_memory(memory_context)}\n\n"
        f"RETRIEVAL CONFIDENCE (computed, use this as confidence_score): {retrieval_confidence}"
    )


def build_claims_user_message(query, signals, memory_context) -> str:
    return (
        f"CLIENT QUERY:\n{query}\n\n"
        f"RETRIEVED HAZRA SIGNALS:\n{format_signals(signals)}\n\n"
        f"CLIENT MEMORY CONTEXT:\n{format_memory(memory_context)}"
    )
