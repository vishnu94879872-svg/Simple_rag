"""Settings, storage backend, privacy, and security view."""

import streamlit as st

from config import settings
from observability.storage import storage
from app.ui import page_header, pill, verdict


def _render_supabase_panel() -> None:
    status = storage.get_status()
    configured = settings.supabase_configured
    is_supabase = storage.backend_name == "supabase"

    if is_supabase and not status.get("degraded"):
        verdict(
            "Supabase connected",
            f"Project {status.get('project_host') or 'unknown'} · table {status.get('table')}. "
            "Observability records are persisted remotely.",
            tone="success",
        )
    elif is_supabase:
        verdict(
            "Supabase degraded",
            f"The last operation fell back to local files. {status.get('last_error') or 'Unknown error.'}",
            tone="warning",
        )
    elif configured:
        verdict(
            "Supabase configured, local fallback active",
            f"{status.get('last_error') or 'Initialization failed.'} Check server logs, then restart Streamlit.",
            tone="warning",
        )
    else:
        verdict(
            "Local file storage",
            "Add SUPABASE_URL and SUPABASE_KEY to .env, run scripts/supabase_schema.sql, then restart Streamlit.",
            tone="info",
        )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Active backend", "Supabase" if is_supabase else "Local files")
    with m2:
        st.metric("Configured", "Yes" if configured else "No")
    with m3:
        st.metric("Fallback operations", status.get("fallback_ops", 0))
    with m4:
        st.metric("Service key", "Hidden" if configured else "Not set")

    left, right = st.columns(2)
    with left:
        st.text_input("Project host", value=status.get("project_host") or "—", disabled=True)
        st.text_input("Records table", value=status.get("table") or settings.supabase_table, disabled=True)
    with right:
        st.text_input("Local fallback directory", value=status.get("local_dir") or str(settings.observability_dir), disabled=True)
        st.text_input(
            "Connection state",
            value=("connected" if status.get("connected") else "degraded" if status.get("degraded") else "not checked"),
            disabled=True,
        )

    if status.get("last_error_at") and status.get("degraded"):
        st.caption(f"Last storage error: {status['last_error_at']} · {status.get('last_error')}")

    st.divider()
    st.markdown("#### Connection and backfill")
    st.caption(
        "Credentials are read server-side from .env. The service key is never rendered in the UI. "
        "Restart Streamlit after changing .env."
    )

    action_left, action_right = st.columns(2)
    with action_left:
        if st.button(
            "Test connection",
            width="stretch",
            disabled=not (configured and hasattr(storage, "check_connection")),
        ):
            with st.spinner("Pinging Supabase PostgREST..."):
                result = storage.check_connection()
            if result.get("connected"):
                st.success(f"Connected to {result.get('project_host')} · table `{result.get('table')}`.")
            else:
                st.error(result.get("last_error") or "Connection failed.")

    with action_right:
        if st.button(
            "Push local records to Supabase",
            width="stretch",
            disabled=not (is_supabase and hasattr(storage, "push_local_records")),
            help="Idempotently upserts every local job, trace, eval run, benchmark, experiment, and investigation.",
        ):
            with st.spinner("Backfilling local observability records..."):
                try:
                    counts = storage.push_local_records()
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.success(
                        "Backfill complete · "
                        + " · ".join(f"{name}: {count}" for name, count in counts.items())
                    )

    if not configured:
        st.markdown("#### Local setup")
        st.code(
            "SUPABASE_URL=https://your-project.supabase.co\n"
            "SUPABASE_KEY=your_service_role_key\n"
            "STORAGE_BACKEND=auto\n\n"
            "# then run scripts/supabase_schema.sql in the Supabase SQL editor",
            language="bash",
        )


def render_settings(pipeline):
    """Render Settings, storage backend, and security configuration."""
    page_header(
        "Module · Configuration",
        "Settings & Security",
        "Inspect runtime configuration, manage the Supabase observability backend, and review the production threat model.",
    )

    tab_runtime, tab_storage, tab_security = st.tabs(["Runtime", "Supabase & Storage", "Threat Model"])

    with tab_runtime:
        st.markdown("### Model and provider")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("LLM provider", value=settings.llm_provider, disabled=True)
            st.text_input("LLM model", value=settings.llm_model, disabled=True)
            st.text_input("Vision model", value=settings.vision_model, disabled=True)
        with c2:
            st.text_input("Embedding model", value=settings.embedding_model, disabled=True)
            st.text_input("Reranker model", value=settings.reranker_model, disabled=True)
            key_raw = settings.llm_api_key
            masked_key = f"{key_raw[:4]}...{key_raw[-4:]}" if len(key_raw) > 8 else ("Not set · offline fallback" if not key_raw else "********")
            st.text_input("API key status", value=masked_key, disabled=True, help="API keys are never displayed in plaintext.")

        st.divider()
        st.markdown("### Storage paths")
        st.text_input("ChromaDB persistence", value=str(settings.chroma_persist_dir), disabled=True)
        st.text_input("BM25 persistence", value=str(settings.bm25_persist_dir), disabled=True)
        st.text_input("Parent store directory", value=str(settings.parent_store_dir), disabled=True)
        st.text_input("Observability directory", value=str(settings.observability_dir), disabled=True)

        st.divider()
        st.markdown("### Session")
        col_clear, _ = st.columns([1, 2])
        with col_clear:
            if st.button("Clear active chat conversation", width="stretch"):
                st.session_state.messages = []
                st.success("Chat conversation cleared.")
                st.rerun()

    with tab_storage:
        _render_supabase_panel()

    with tab_security:
        st.markdown("### Production security architecture & threat model")

        st.markdown("**1 · Untrusted document text defense**")
        st.markdown(
            "Uploaded PDFs may contain prompt-injection sequences. Retrieved text is treated strictly as passive data, "
            "delimited in explicit context blocks, and the system prompt instructs the model to ignore instructions inside quotes."
        )

        st.markdown("**2 · Secret redaction & key protection**")
        st.markdown(
            "The mask_secrets filter parses every trace, error, and step payload before serialization, redacting API keys "
            "and bearer tokens. Supabase credentials are read server-side only and never rendered in the UI."
        )

        st.markdown("**3 · Path traversal & upload validation**")
        st.markdown(
            "Uploads pass through sanitize_filename and are restricted to validated PDF content stored under data/books/."
        )

        st.markdown("**4 · Index isolation & idempotent upserts**")
        st.markdown(
            "Chunks use deterministic content IDs, so re-indexing updates records in place. Experiments run against isolated "
            "sandboxes and never touch the production index."
        )

        st.divider()
        st.markdown(
            f"{pill('Service key hidden', 'success')} {pill('Prompt injection bounded', 'accent')} {pill('Trace redaction on', 'accent')}",
            unsafe_allow_html=True,
        )
