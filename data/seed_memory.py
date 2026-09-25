"""Three past Haleon decisions, loaded into client_memory at startup."""

SEED_MEMORY = [
    {
        "decision_id": "MEM-HAL-2024-001",
        "client_id": "haleon",
        "decision_type": "brief_approved",
        "query_summary": "What is the whitespace in GLP-1 nutrition for our vitamins portfolio",
        "direction": "Muscle preservation positioning for GLP-1 users",
        "rejection_reason": None,
        "created_at": "2024-03-15",
    },
    {
        "decision_id": "MEM-HAL-2024-002",
        "client_id": "haleon",
        "decision_type": "claim_rejected",
        "query_summary": "Draft marketing claims for GLP-1 muscle product",
        "direction": "Clinically proven to build muscle on GLP-1 medication",
        "rejection_reason": (
            "Legal rejected: clinically proven language requires cited clinical study. "
            "Use supports or designed to instead."
        ),
        "created_at": "2024-03-22",
    },
    {
        "decision_id": "MEM-HAL-2024-003",
        "client_id": "haleon",
        "decision_type": "claim_selected",
        "query_summary": "Draft marketing claims for GLP-1 muscle product",
        "direction": "Supports muscle maintenance during weight management with GLP-1 medications",
        "rejection_reason": None,
        "created_at": "2024-03-22",
    },
]
