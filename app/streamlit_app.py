"""BookRAG Reliability Lab — Advanced RAG Engineering Platform.

Production-oriented RAG debugging, evaluation, and observability dashboard.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is on sys.path so that top-level modules
# (config, generation, retrieval, ingest, observability) are importable
# regardless of the directory Streamlit is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import settings
from generation.qa_chain import BookRAGPipeline
from observability.storage import storage
from app.ui import inject_theme, sidebar_brand, sidebar_nav_label, sidebar_status, storage_banner

# View Modules
from app.views.overview_view import render_overview
from app.views.chat_view import render_chat
from app.views.ingestion_view import render_ingestion_monitor
from app.views.inspector_view import render_chunk_inspector
from app.views.retrieval_view import render_retrieval_playground
from app.views.trace_view import render_trace_viewer
from app.views.citation_view import render_citation_verification
from app.views.evaluation_view import render_evaluation_lab
from app.views.diagnosis_view import render_root_cause_assistant
from app.views.experiments_view import render_experiment_workbench
from app.views.settings_view import render_settings

st.set_page_config(
    page_title="BookRAG Reliability Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()


@st.cache_resource(show_spinner="Loading embedding and reranking engines...")
def get_pipeline():
    """Cache the BookRAG pipeline to avoid re-loading sentence transformer weights."""
    return BookRAGPipeline()


pipeline = get_pipeline()

NAV_ITEMS = {
    "Overview": render_overview,
    "Book Tutor": render_chat,
    "Ingestion Monitor": render_ingestion_monitor,
    "Chunk Inspector": render_chunk_inspector,
    "Retrieval Playground": render_retrieval_playground,
    "Trace Viewer": render_trace_viewer,
    "Citation Verification": render_citation_verification,
    "Evaluation Lab": render_evaluation_lab,
    "Root-Cause Assistant": render_root_cause_assistant,
    "Chunking Experiments": render_experiment_workbench,
    "Settings & Security": render_settings,
}

# --- Sidebar ---
with st.sidebar:
    sidebar_brand()
    st.divider()
    sidebar_nav_label("Workspace")

    selected_module = st.radio(
        "Navigation",
        options=list(NAV_ITEMS.keys()),
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    sidebar_nav_label("Runtime")
    sidebar_status(
        storage.get_status(),
        vector_count=pipeline.retrieval_engine.vector_store.count(),
        bm25_count=pipeline.retrieval_engine.bm25_store.count(),
        model=settings.llm_model,
    )
    st.markdown(
        '<div style="font-family:var(--br-mono);font-size:8.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--br-text-dim);margin-top:8px;">'
        "Ingest → Retrieve → Rerank → Audit"
        "</div>",
        unsafe_allow_html=True,
    )

# --- Main router ---
storage_banner(storage.get_status())
if selected_module in {"Overview", "Book Tutor", "Evaluation Lab", "Settings & Security"}:
    NAV_ITEMS[selected_module](pipeline)
else:
    NAV_ITEMS[selected_module]()
