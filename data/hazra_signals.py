"""The five hardcoded HAZRA signals. Verbatim from the CREWASIS PRD. Do not change."""

HAZRA_SIGNALS = [
    {
        "signal_id": "HAZ-2024-GLP1-001",
        "signal_text": "Consumer concern about muscle loss on GLP-1 medications is climbing fast. Volume up 340 percent in 90 days. Top consumer language: preserve muscle, maintain strength, protein while on Ozempic.",
        "source_type": "consumer_review_aggregate",
        "source_platforms": ["Reddit r/Ozempic", "Amazon supplement reviews", "WebMD patient forums"],
        "date_range": "2024-01-01 to 2024-03-31",
        "hazra_momentum_score": 0.91,
        "hazra_confidence": 0.87,
    },
    {
        "signal_id": "HAZ-2024-GLP1-002",
        "signal_text": "No major supplement brand has launched a clinical-language muscle preservation claim specifically for GLP-1 users. Clear whitespace in the category.",
        "source_type": "competitive_gap_analysis",
        "source_platforms": ["Retail shelf audit", "Competitor press release monitoring", "FDC product database"],
        "date_range": "2024-01-01 to 2024-03-31",
        "hazra_momentum_score": 0.87,
        "hazra_confidence": 0.82,
    },
    {
        "signal_id": "HAZ-2024-GLP1-003",
        "signal_text": "Historical category response time: GLP-1 nutrition moved from first detectable consumer signal to first major competitor launch in approximately 24 months. Current signal age 8 months. Urgency tier: act this year.",
        "source_type": "category_timing_record",
        "source_platforms": ["BusinessWire press release February 8 2024 Herbalife GLP-1 Nutrition Companion"],
        "date_range": "2022-01-01 to 2024-02-08",
        "hazra_momentum_score": 0.79,
        "hazra_confidence": 0.94,
    },
    {
        "signal_id": "HAZ-2024-GLP1-004",
        "signal_text": "Consumer language strongly favors maintenance and preservation framing over building or growth. Top phrases: support muscle maintenance, preserve lean muscle, maintain strength on medication.",
        "source_type": "consumer_language_analysis",
        "source_platforms": ["Reddit r/loseit", "MyFitnessPal forums", "TikTok comment analysis"],
        "date_range": "2024-02-01 to 2024-04-30",
        "hazra_momentum_score": 0.84,
        "hazra_confidence": 0.79,
    },
    {
        "signal_id": "HAZ-2024-GLP1-005",
        "signal_text": "Claims using clinically proven language have a high regulatory rejection rate in this category. Safe language patterns: supports, may help, designed to. Risky patterns: proven, clinically proven, prevents.",
        "source_type": "regulatory_risk_analysis",
        "source_platforms": ["FDA warning letter database", "FTC enforcement actions 2022-2024"],
        "date_range": "2022-01-01 to 2024-03-31",
        "hazra_momentum_score": 0.76,
        "hazra_confidence": 0.91,
    },
]
