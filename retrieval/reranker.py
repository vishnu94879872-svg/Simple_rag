"""Neural reranker using CrossEncoder (BAAI/bge-reranker-base)."""

import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

from config import settings

logger = logging.getLogger(__name__)


class BGEReranker:
    """Wrapper for CrossEncoder reranking model."""

    _instance = None
    _model = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.reranker_model
        if BGEReranker._model is None:
            logger.info("Loading CrossEncoder reranker model: %s", self.model_name)
            BGEReranker._model = CrossEncoder(self.model_name)
        self.model = BGEReranker._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Score (query, chunk_content) pairs and return top_k candidates sorted by reranker score.
        """
        if not candidates:
            return []

        if len(candidates) <= 1:
            candidates[0]["rerank_score"] = 1.0
            return candidates[:top_k]

        pairs = [[query, c["content"]] for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach rerank scores
        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = float(score)

        # Sort descending by rerank score
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
