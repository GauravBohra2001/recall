"""Print retrieval and memory results for the three demo queries, offline.

Run after installing dependencies. It needs no API keys: it touches retrieval only.
Use the output to set PRECEDENT_SIMILARITY_THRESHOLD and CONFLICT_SIMILARITY_THRESHOLD
in agents/memory_agent.py so that:

    Query 1 -> precedent_found (yellow banner, no conflict gate)
    Query 2 -> claims path, MEM-HAL-2024-002 visible to the claims agent
    Query 3 -> conflict_found (conflict gate), then the secondary retrieval loop

    python -m scripts.tune_thresholds
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.memory_agent import (  # noqa: E402
    CONFLICT_SIMILARITY_THRESHOLD,
    POSITIONING_CONFLICT_FLOOR,
    PRECEDENT_SIMILARITY_THRESHOLD,
    classify_finding,
)
from agents.strategy_agent import GAP_SOURCE_TYPE, PRIMARY_TOP_K  # noqa: E402
from data.hazra_signals import HAZRA_SIGNALS  # noqa: E402
from db.database import get_connection, init_db  # noqa: E402
from rag.retrieval import (  # noqa: E402
    compute_confidence,
    compute_confidence_prd_literal,
    hybrid_search,
    memory_search,
)
from rag.router import is_claims_query, is_whitespace_query, route_query  # noqa: E402

QUERIES = [
    "What should we do about the muscle health opportunity for GLP-1 customers?",
    "Draft a marketing claim for our GLP-1 muscle preservation product.",
    "What is the competitive whitespace in GLP-1 nutrition for Haleon?",
]


def main():
    conn = get_connection()
    init_db(conn)
    print(f"thresholds: precedent >= {PRECEDENT_SIMILARITY_THRESHOLD}, "
          f"conflict >= {CONFLICT_SIMILARITY_THRESHOLD}, "
          f"positioning-conflict floor {POSITIONING_CONFLICT_FLOOR}")

    for i, query in enumerate(QUERIES, start=1):
        print(f"\n{'=' * 78}\nQuery {i}: {query}")
        print(f"route={route_query(query)}  claims_path={is_claims_query(query)}  "
              f"whitespace={is_whitespace_query(query)}")

        signals = hybrid_search(query, HAZRA_SIGNALS, top_k=PRIMARY_TOP_K)
        types = {s["source_type"] for s in signals}
        print(f"retrieved: {[s['signal_id'] for s in signals]}")
        print(f"confidence (weighted, used): {compute_confidence(signals)}   "
              f"(PRD literal: {compute_confidence_prd_literal(signals)})")
        print(f"secondary retrieval would trigger: "
              f"{is_whitespace_query(query) and GAP_SOURCE_TYPE not in types}")

        ctx = memory_search(query, "haleon", conn, top_k=5, min_similarity=0.0)
        print("memory similarity:")
        for m in ctx:
            print(f"  {m['similarity_score']:.3f}  {m['decision_type']:<15} "
                  f"{m['decision_id']}  {m['direction'][:58]}")
        finding, driver = classify_finding(ctx, query)
        print(f"=> memory_finding: {finding}" + (f" via {driver['decision_id']}" if driver else ""))


if __name__ == "__main__":
    main()
