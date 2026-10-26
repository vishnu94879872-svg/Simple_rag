"""BookRAG Reliability Lab — Advanced RAG Engineering Platform & Book Tutor.

Production-oriented RAG platform supporting both reader-facing Book Tutor
and engineering observability lab with secure role-based access.
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

# Reader Portal Views
from app.views.chat_view import render_chat
from app.views.user_upload_view import render_user_upload
from app.views.support_view import render_support_view

# Admin Observability Views
from app.views.overview_view import render_overview
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
    page_title="Book Tutor · AI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()


@st.cache_resource(show_spinner="Loading embedding and reranking engines...")
def get_pipeline():
    """Cache the BookRAG pipeline to avoid re-loading sentence transformer weights."""
    return BookRAGPipeline()


pipeline = get_pipeline()

# Session State for Role Separation
if "portal" not in st.session_state:
    st.session_state.portal = "user"
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False
if "user_nav" not in st.session_state:
    st.session_state.user_nav = "Book Tutor"
if "admin_nav" not in st.session_state:
    st.session_state.admin_nav = "Overview"

USER_NAV_ITEMS = {
    "Book Tutor": render_chat,
    "Upload Book": render_user_upload,
    "Support & Reports": render_support_view,
}

ADMIN_NAV_ITEMS = {
    "Overview": render_overview,
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

# --- Sidebar Layout ---
with st.sidebar:
    if st.session_state.portal == "admin":
        if not st.session_state.admin_authenticated:
            st.session_state.portal = "user"
            st.rerun()

        sidebar_brand(title="BookRAG", subtitle="Admin Reliability Lab", mark="🛡️")
        st.divider()

        col_portal, col_logout = st.columns([1, 1])
        with col_portal:
            if st.button("🎓 Reader Portal", use_container_width=True):
                st.session_state.portal = "user"
                st.rerun()
        with col_logout:
            if st.button("🚪 Log Out", use_container_width=True):
                st.session_state.admin_authenticated = False
                st.session_state.portal = "user"
                st.rerun()

        st.divider()
        sidebar_nav_label("Observability Lab")

        selected_module = st.radio(
            "Admin Navigation",
            options=list(ADMIN_NAV_ITEMS.keys()),
            key="admin_nav",
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

    else:
        # User / Reader Portal
        sidebar_brand(title="Book Tutor", subtitle="Interactive Reader", mark="📚")
        st.divider()
        sidebar_nav_label("Reader Workspace")

        selected_module = st.radio(
            "Reader Navigation",
            options=list(USER_NAV_ITEMS.keys()),
            key="user_nav",
            label_visibility="collapsed",
        )

        st.divider()
        # Admin Portal Entry
        with st.expander("🔐 Admin Access", expanded=False):
            if st.session_state.admin_authenticated:
                st.caption("You are currently authenticated as Admin.")
                if st.button("Open Admin Lab 🛡️", type="primary", use_container_width=True):
                    st.session_state.portal = "admin"
                    st.rerun()
            else:
                st.caption("Sign in to access engineering observability & telemetry.")
                login_user = st.text_input("Username", key="login_username_input")
                login_pass = st.text_input("Password", type="password", key="login_password_input")
                if st.button("Unlock Admin Lab", type="primary", use_container_width=True):
                    if (
                        login_user.strip() == settings.admin_username
                        and login_pass.strip() == settings.admin_password
                    ):
                        st.session_state.admin_authenticated = True
                        st.session_state.portal = "admin"
                        st.session_state.admin_nav = "Overview"
                        st.rerun()
                    else:
                        st.error("Invalid admin credentials.")

# --- Main Router ---
storage_banner(storage.get_status())

if st.session_state.portal == "user":
    if selected_module == "Book Tutor":
        render_chat(pipeline)
    elif selected_module == "Upload Book":
        render_user_upload(pipeline)
    elif selected_module == "Support & Reports":
        render_support_view(pipeline)
else:
    # Admin Portal Router
    if selected_module in {"Overview", "Evaluation Lab", "Settings & Security"}:
        ADMIN_NAV_ITEMS[selected_module](pipeline)
    else:
        ADMIN_NAV_ITEMS[selected_module]()
