"""Private factory module for store selection.

Kept separate from retrieval/__init__.py to avoid circular imports
(engine.py and indexer.py are imported by __init__.py, so they cannot
import from retrieval directly).
"""

import logging
from config import settings

logger = logging.getLogger(__name__)

_USE_SUPABASE = (
    settings.storage_backend == "supabase"
    or (settings.storage_backend == "auto" and settings.supabase_configured)
)


def get_vector_store():
    """Return SupabaseVectorStore when Supabase is configured, else ChromaVectorStore."""
    if _USE_SUPABASE:
        from retrieval.supabase_vector_store import SupabaseVectorStore
        logger.info("Using SupabaseVectorStore (cloud backend)")
        return SupabaseVectorStore()
    from retrieval.vector_store import ChromaVectorStore
    logger.info("Using ChromaVectorStore (local backend)")
    return ChromaVectorStore()


def get_bm25_store():
    """Return SupabaseBM25Store when Supabase is configured, else BM25Store."""
    if _USE_SUPABASE:
        from retrieval.supabase_bm25_store import SupabaseBM25Store
        logger.info("Using SupabaseBM25Store (cloud backend)")
        return SupabaseBM25Store()
    from retrieval.bm25_store import BM25Store
    logger.info("Using BM25Store (local backend)")
    return BM25Store()
