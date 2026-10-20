"""BookRAG Reliability Lab — Advanced RAG Engineering Platform.

Production-oriented RAG debugging, evaluation, and observability dashboard.
"""

import sys
import streamlit as st
from pathlib import Path

# Ensure the project root is on sys.path so that top-level modules
# (config, generation, retrieval, ingest, observability) are importable
# regardless of the directory Streamlit is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import settings
from generation.qa_chain import BookRAGPipeline

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
    initial_sidebar_state="expanded"
)

# Custom CSS for dark-mode developer dashboard
st.markdown("""
<style>
    /* Dark charcoal developer theme */
    .stApp {
        background-color: #121214;
        color: #f8fafc;
    }
    
    /* Header typography */
    .lab-header {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin-bottom: 0.1rem;
    }
    .lab-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    
    /* Source badges */
    .source-tag {
        display: inline-block;
        background-color: #1e3a8a33;
        color: #93c5fd;
        border: 1px solid #3b82f644;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-top: 0.4rem;
    }
    
    /* Chunk card and text wrap fixes for horizontal overflow */
    .chunk-card {
        background-color: #18181b;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 0.85rem;
        margin-bottom: 0.65rem;
        word-break: break-word;
        overflow-wrap: break-word;
    }
    
    /* Ensure code blocks and pre elements wrap gracefully without breaking viewport */
    pre, code {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0d0d10;
        border-right: 1px solid #27272a;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing BookRAG neural embedding & reranking engines...")
def get_pipeline():
    """Cache the BookRAG pipeline to avoid re-loading sentence transformer weights."""
    return BookRAGPipeline()


pipeline = get_pipeline()

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown('<div class="lab-header">🛡️ BookRAG Lab</div>', unsafe_allow_html=True)
    st.caption("Advanced RAG Reliability & Observability")
    st.divider()

    selected_module = st.radio(
        "Navigation",
        options=[
            "📊 Overview",
            "💬 Book Tutor Chat",
            "📥 Ingestion Monitor",
            "🔍 Chunk Inspector",
            "🎯 Retrieval Playground",
            "⏱️ Trace Viewer",
            "🛡️ Citation Verification",
            "🧪 Evaluation Lab",
            "🩺 Root-Cause Assistant",
            "🔬 Chunking Experiments",
            "⚙️ Settings & Security"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()
    # Live status footer
    chroma_count = pipeline.retrieval_engine.vector_store.count()
    bm25_count = pipeline.retrieval_engine.bm25_store.count()
    st.caption(f"**Chroma Vectors:** `{chroma_count}`")
    st.caption(f"**BM25 Documents:** `{bm25_count}`")
    st.caption(f"**LLM Model:** `{settings.llm_model}`")


# --- Main Router ---
if selected_module == "📊 Overview":
    render_overview(pipeline)
elif selected_module == "💬 Book Tutor Chat":
    render_chat(pipeline)
elif selected_module == "📥 Ingestion Monitor":
    render_ingestion_monitor()
elif selected_module == "🔍 Chunk Inspector":
    render_chunk_inspector()
elif selected_module == "🎯 Retrieval Playground":
    render_retrieval_playground()
elif selected_module == "⏱️ Trace Viewer":
    render_trace_viewer()
elif selected_module == "🛡️ Citation Verification":
    render_citation_verification()
elif selected_module == "🧪 Evaluation Lab":
    render_evaluation_lab(pipeline)
elif selected_module == "🩺 Root-Cause Assistant":
    render_root_cause_assistant()
elif selected_module == "🔬 Chunking Experiments":
    render_experiment_workbench()
elif selected_module == "⚙️ Settings & Security":
    render_settings(pipeline)
