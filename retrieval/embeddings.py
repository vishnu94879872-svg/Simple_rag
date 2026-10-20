"""Embedding model manager using sentence-transformers."""

import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Singleton-style wrapper for SentenceTransformer embeddings."""

    _instance = None
    _model = None

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        if EmbeddingManager._model is None:
            logger.info("Loading SentenceTransformer embedding model: %s", self.model_name)
            EmbeddingManager._model = SentenceTransformer(self.model_name)
        self.model = EmbeddingManager._model

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Compute dense vector embeddings for a list of document strings."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute embedding for a single query string."""
        emb = self.model.encode(text, normalize_embeddings=True)
        return emb.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimensionality."""
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()
