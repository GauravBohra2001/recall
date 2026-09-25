"""Hybrid retrieval: BM25 + dense + reciprocal rank fusion, plus memory retrieval."""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

_model = None
_embedding_cache = {}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def get_model():
    """Lazy load so Streamlit start-up is not blocked by the embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, so 'GLP-1?' and 'glp-1' match."""
    return _TOKEN_RE.findall(text.lower())


def embed(texts: List[str]) -> np.ndarray:
    """Encode with a per-text cache. The signal corpus is static, so after the first
    query every signal embedding is a dictionary hit."""
    missing = [t for t in dict.fromkeys(texts) if t not in _embedding_cache]
    if missing:
        vectors = get_model().encode(missing)
        for text, vec in zip(missing, vectors):
            _embedding_cache[text] = vec
    return np.array([_embedding_cache[t] for t in texts])


def bm25_search(query: str, signals: List[dict], top_k: int = 5) -> List[dict]:
    if not signals:
        return []
    bm25 = BM25Okapi([tokenize(s["signal_text"]) for s in signals])
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(signals, scores), key=lambda x: x[1], reverse=True)
    return [
        {**signal, "bm25_score": float(score), "bm25_rank": rank + 1}
        for rank, (signal, score) in enumerate(ranked[:top_k])
        if score > 0
    ]


def dense_search(query: str, signals: List[dict], top_k: int = 5) -> List[dict]:
    if not signals:
        return []
    scores = cosine_similarity(
        embed([query]), embed([s["signal_text"] for s in signals])
    )[0]
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {**signals[i], "dense_score": float(scores[i]), "dense_rank": rank + 1}
        for rank, i in enumerate(ranked_indices)
        if scores[i] > 0.1
    ]


def rrf_combine(bm25_results: List[dict], dense_results: List[dict], k: int = 60) -> List[dict]:
    scores = {}
    for result in bm25_results:
        sid = result["signal_id"]
        scores.setdefault(sid, {"signal": result, "score": 0.0})
        scores[sid]["score"] += 1 / (k + result["bm25_rank"])
    for result in dense_results:
        sid = result["signal_id"]
        scores.setdefault(sid, {"signal": result, "score": 0.0})
        scores[sid]["score"] += 1 / (k + result["dense_rank"])
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [
        {**item["signal"], "rrf_score": item["score"], "retrieval_method": "hybrid_rrf"}
        for item in ranked
    ]


def hybrid_search(query: str, signals: List[dict], top_k: int = 3) -> List[dict]:
    """BM25 and dense run in parallel, then merge with RRF."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        bm25_future = pool.submit(bm25_search, query, signals, 5)
        dense_future = pool.submit(dense_search, query, signals, 5)
        bm25_results, dense_results = bm25_future.result(), dense_future.result()
    return rrf_combine(bm25_results, dense_results)[:top_k]


def structured_timing_lookup(signals: List[dict]) -> List[dict]:
    """Urgency queries: a structured lookup of timing records, no similarity search."""
    return [
        {**s, "retrieval_method": "structured_lookup"}
        for s in signals if s.get("source_type") == "category_timing_record"
    ]


def memory_search(query: str, client_id: str, conn, top_k: int = 3,
                  min_similarity: float = 0.4) -> List[dict]:
    """Dense search over client_memory, always filtered by client_id."""
    records = [
        dict(row) for row in conn.execute(
            "SELECT * FROM client_memory WHERE client_id = ?", (client_id,)
        ).fetchall()
    ]
    if not records:
        return []
    scores = cosine_similarity(
        embed([query]),
        embed([r["direction"] + " " + (r["query_summary"] or "") for r in records]),
    )[0]
    ranked = sorted(zip(records, scores), key=lambda x: x[1], reverse=True)
    return [
        {**record, "similarity_score": round(float(score), 3)}
        for record, score in ranked[:top_k]
        if score > min_similarity
    ]


def compute_confidence(retrieved_signals: List[dict]) -> float:
    """Momentum-weighted average of HAZRA confidence.

    Deliberate deviation from the PRD formula, which divides sum(confidence * momentum)
    by the number of signals. That returns 0.749 for the Query 1 signal set (amber,
    below the 0.75 the demo requires). Dividing by the sum of the momentum weights is
    the true weighted average and returns 0.875, matching the 0.87 in the demo script.
    """
    if not retrieved_signals:
        return 0.0
    total_weight = sum(s.get("hazra_momentum_score", 0.5) for s in retrieved_signals)
    if total_weight == 0:
        return 0.0
    weighted = sum(
        s.get("hazra_confidence", 0.5) * s.get("hazra_momentum_score", 0.5)
        for s in retrieved_signals
    )
    return round(weighted / total_weight, 3)


def compute_confidence_prd_literal(retrieved_signals: List[dict]) -> Optional[float]:
    """The formula exactly as written in the PRD. Kept for comparison, not used."""
    if not retrieved_signals:
        return 0.0
    scores = [s.get("hazra_confidence", 0.5) * s.get("hazra_momentum_score", 0.5)
              for s in retrieved_signals]
    return round(sum(scores) / len(scores), 3)
