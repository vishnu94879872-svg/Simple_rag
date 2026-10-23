"""BookRAG Retrieval Package.

Factory functions ``get_vector_store()`` and ``get_bm25_store()`` automatically
select the correct backend (Supabase for cloud, Chroma/BM25 for local) based on
the STORAGE_BACKEND setting and whether SUPABASE_URL / SUPABASE_KEY are configured.
"""

from retrieval.embeddings import EmbeddingManager
from retrieval.vector_store import ChromaVectorStore
from retrieval.bm25_store import BM25Store
from retrieval.indexer import BookIndexer
from retrieval.query_rewriter import QueryRewriter
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import BGEReranker
from retrieval.context_expansion import ContextExpander
from retrieval.engine import RetrievalEngine, RetrievalResult
from retrieval._store_factory import get_vector_store, get_bm25_store


__all__ = [
    "EmbeddingManager",
    "ChromaVectorStore",
    "BM25Store",
    "BookIndexer",
    "QueryRewriter",
    "reciprocal_rank_fusion",
    "BGEReranker",
    "ContextExpander",
    "RetrievalEngine",
    "RetrievalResult",
    "get_vector_store",
    "get_bm25_store",
]
