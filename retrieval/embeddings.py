"""Embedding clients for local SentenceTransformers and Superlinked SIE."""

import json
import logging
import math
from typing import List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


def _normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("Embedding service returned a zero-length vector")
    return [value / norm for value in vector]


class EmbeddingManager:
    """Provide consistently normalized embeddings from the configured backend."""

    _models = {}

    def __init__(self, model_name: Optional[str] = None):
        self.backend = settings.embedding_backend
        self.model_name = model_name or settings.embedding_model
        if self.backend == "local":
            if self.model_name not in self._models:
                logger.info("Loading SentenceTransformer embedding model: %s", self.model_name)
                self._models[self.model_name] = SentenceTransformer(self.model_name)
            self.model = self._models[self.model_name]
        elif self.backend == "sie":
            self.model = None
        else:
            raise ValueError(f"Unsupported EMBEDDING_BACKEND '{self.backend}'. Use 'sie' or 'local'.")

    def _embed_sie(self, texts: List[str]) -> List[List[float]]:
        payload = json.dumps({"model": self.model_name, "input": texts}).encode("utf-8")
        request = Request(
            settings.sie_embeddings_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"SIE embedding request failed ({exc.code}): {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Cannot reach SIE at {settings.sie_embeddings_url}: {exc}") from exc

        try:
            rows = result["data"]
            if all("index" in row for row in rows):
                rows = sorted(rows, key=lambda row: row["index"])
            vectors = [_normalize([float(value) for value in row["embedding"]]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("SIE returned an invalid embeddings response") from exc
        if len(vectors) != len(texts) or not vectors or any(len(v) != len(vectors[0]) for v in vectors):
            raise RuntimeError("SIE returned a different number of embeddings than requested")
        return vectors

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Compute normalized dense embeddings for document strings."""
        if not texts:
            return []
        if self.backend == "local":
            embeddings = self.model.encode(
                texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
            )
            return embeddings.tolist()
        batch_size = max(1, batch_size)
        result = []
        for start in range(0, len(texts), batch_size):
            result.extend(self._embed_sie(texts[start:start + batch_size]))
        return result

    def embed_query(self, text: str) -> List[float]:
        """Compute a normalized embedding for one query."""
        if self.backend == "local":
            return self.model.encode(text, normalize_embeddings=True).tolist()
        return self._embed_sie([text])[0]

    @property
    def dimension(self) -> int:
        """Return the vector width, querying SIE lazily when needed."""
        if self.backend == "local":
            if hasattr(self.model, "get_embedding_dimension"):
                return self.model.get_embedding_dimension()
            return self.model.get_sentence_embedding_dimension()
        return len(self.embed_query("embedding dimension probe"))
