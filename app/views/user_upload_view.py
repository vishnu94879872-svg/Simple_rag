"""Reader Portal — Book Library & Upload View.

Provides a clean, friendly interface for users to upload PDF books,
monitor live ingestion progress, and see their active bookshelf.
"""

from pathlib import Path
import streamlit as st

from config import settings
from observability.ingestion_monitor import ingestion_monitor
from observability.models import StageStatus
from app.ui import empty_state, escape, page_header, pill


def render_user_upload(pipeline):
    """Render the reader-facing book upload and bookshelf view."""
    page_header(
        "Reader Workspace · Library",
        "Upload & Manage Books",
        "Upload PDF books to your personal library, track processing progress, and begin learning with Book Tutor.",
    )

    col_upload, col_bookshelf = st.columns([1.2, 1.8], gap="large")

    with col_upload:
        st.markdown(
            """
            <div style="background:var(--br-surface);border:1px solid var(--br-border);border-radius:12px;padding:20px;margin-bottom:20px;">
                <div style="font-size:16px;font-weight:600;color:var(--br-text);margin-bottom:6px;">Upload a New Book</div>
                <div style="font-size:13px;color:var(--br-text-muted);margin-bottom:16px;">
                    Upload any PDF textbook, paper, or manual to index its chapters, text, and tables for interactive tutoring.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Select a PDF book",
            type=["pdf"],
            help="Upload textbooks, documentation, or study guides in PDF format.",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            default_title = Path(uploaded_file.name).stem.replace("_", " ").title()
            custom_title = st.text_input("Book Title", value=default_title, help="Name used to identify this book in Book Tutor")

            if st.button("🚀 Upload & Index Book", type="primary", use_container_width=True):
                save_path = settings.data_dir / "books" / uploaded_file.name
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.status("📖 Processing and indexing book...", expanded=True) as status:
                    st.write("📄 **Step 1/4:** Parsing PDF layout, headers, and pages...")
                    job = ingestion_monitor.run_monitored_ingestion(save_path)
                    st.write("✂️ **Step 2/4:** Cleaning text and generating smart chunks...")
                    st.write("🧠 **Step 3/4:** Creating dense vector embeddings (BGE-small)...")
                    st.write("☁️ **Step 4/4:** Syncing index to Supabase pgvector & BM25...")

                    st.cache_resource.clear()

                    if job.overall_status in (StageStatus.COMPLETED, StageStatus.WARNING):
                        status.update(
                            label=f"✅ '{custom_title}' is indexed and ready to tutor!",
                            state="complete",
                            expanded=False,
                        )
                        st.success(
                            f"**Success!** Indexed **{job.page_count} pages** into **{job.chunk_count} chunks**. "
                            f"You can now ask questions about this book in the Book Tutor."
                        )
                        col_chat_btn, _ = st.columns([1, 1])
                        with col_chat_btn:
                            if st.button("💬 Start Chatting with this Book", type="primary", use_container_width=True):
                                st.session_state.selected_book_focus = custom_title
                                st.session_state.user_nav = "Book Tutor"
                                st.rerun()
                    else:
                        error_msg = job.error_message or "Unknown processing error occurred."
                        status.update(
                            label=f"❌ Ingestion failed for '{uploaded_file.name}'",
                            state="error",
                            expanded=True,
                        )
                        st.error(f"Processing error: {error_msg}")

    with col_bookshelf:
        st.markdown(
            """
            <div style="font-size:16px;font-weight:600;color:var(--br-text);margin-bottom:6px;">Your Active Bookshelf</div>
            <div style="font-size:13px;color:var(--br-text-muted);margin-bottom:16px;">
                Books currently indexed and available for interactive questions:
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            available_books = pipeline.retrieval_engine.bm25_store.get_distinct_books()
        except Exception:
            available_books = []

        if not available_books:
            # Fallback to local files if database empty
            book_files = list(Path(settings.data_dir / "books").glob("*.pdf")) if (settings.data_dir / "books").exists() else []
            available_books = [p.stem.replace("_", " ").title() for p in book_files]

        if available_books:
            for i, book in enumerate(available_books, 1):
                with st.container():
                    st.markdown(
                        f"""
                        <div style="background:var(--br-surface);border:1px solid var(--br-border);border-radius:10px;padding:16px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <div style="font-size:15px;font-weight:600;color:var(--br-text);">📖 {escape(book)}</div>
                                <div style="font-size:12px;color:var(--br-text-dim);margin-top:4px;">
                                    Status: <span style="color:#10b981;font-weight:500;">Indexed & Ready</span> · Grounded in Supabase
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    c_btn, _ = st.columns([1, 2])
                    with c_btn:
                        if st.button(f"Chat with '{book}'", key=f"chat_book_{i}", use_container_width=True):
                            st.session_state.selected_book_focus = book
                            st.session_state.user_nav = "Book Tutor"
                            st.rerun()
        else:
            empty_state(
                "No books indexed yet",
                "Upload a PDF book on the left to start tutoring.",
            )
