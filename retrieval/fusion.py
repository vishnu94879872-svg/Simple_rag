"""Reciprocal Rank Fusion (RRF) for merging dense vector and BM25 sparse search results."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Merge and rank two ranked lists using Reciprocal Rank Fusion (RRF).
    Formula: RRF_score(d) = sum(1 / (rrf_k + rank(d)))
    
    Returns a unified list of merged candidates sorted by descending RRF score.
    """
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Process vector rankings (rank starts at 1)
    for rank, item in enumerate(vector_results, 1):
        doc_id = item["chunk_id"]
        rrf_contrib = 1.0 / (rrf_k + rank)
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "chunk_id": doc_id,
                "content": item["content"],
                "metadata": item["metadata"],
                "vector_rank": rank,
                "vector_score": item.get("score", 0.0),
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": rrf_contrib
            }
        else:
            doc_map[doc_id]["vector_rank"] = rank
            doc_map[doc_id]["vector_score"] = item.get("score", 0.0)
            doc_map[doc_id]["rrf_score"] += rrf_contrib

    # Process BM25 rankings (rank starts at 1)
    for rank, item in enumerate(bm25_results, 1):
        doc_id = item["chunk_id"]
        rrf_contrib = 1.0 / (rrf_k + rank)
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "chunk_id": doc_id,
                "content": item["content"],
                "metadata": item["metadata"],
                "vector_rank": None,
                "vector_score": None,
                "bm25_rank": rank,
                "bm25_score": item.get("score", 0.0),
                "rrf_score": rrf_contrib
            }
        else:
            doc_map[doc_id]["bm25_rank"] = rank
            doc_map[doc_id]["bm25_score"] = item.get("score", 0.0)
            doc_map[doc_id]["rrf_score"] += rrf_contrib

    merged = list(doc_map.values())
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)

    logger.debug(
        "RRF merged %d vector results and %d BM25 results into %d distinct documents",
        len(vector_results),
        len(bm25_results),
        len(merged)
    )
    return merged
