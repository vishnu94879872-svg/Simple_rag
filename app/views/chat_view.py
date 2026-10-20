"""Book Tutor interactive chat view preserving core BookRAG experience with observability enhancements."""

import streamlit as st
from typing import Dict, Any

from config import settings


def render_chat(pipeline):
    """Render Book Tutor Chat interface."""
    st.markdown("## 💬 Book Tutor — Interactive QA")
    st.caption("Ask natural-language questions about indexed books. Every answer is grounded with Chapter and Page citations.")

    # Controls row
    c_strat, c_topk, c_type, c_chap = st.columns([1.5, 1, 1, 1.5])
    with c_strat:
        strategy = st.selectbox(
            "Retrieval Strategy",
            options=["reranked", "hybrid_rrf", "vector_only", "bm25_only"],
            index=0,
            help="Select retrieval pipeline variant"
        )
    with c_topk:
        top_k = st.selectbox("Top-k Chunks", options=[3, 5, 10], index=1)
    with c_type:
        type_filter = st.selectbox(
            "Content Type",
            options=["All", "text", "table", "image", "code", "list"],
            index=0
        )
    with c_chap:
        chapter_filter = st.text_input("Chapter Filter (optional)", value="").strip()

    # Session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown("**Citations:**")
                source_html = " ".join([
                    f'<span class="source-tag">📖 {src["chapter"]} (Page {src["page"]})</span>'
                    for src in msg["sources"]
                ])
                st.markdown(source_html, unsafe_allow_html=True)

            # Audit badge if available
            if msg.get("audit"):
                aud = msg["audit"]
                verdict = getattr(aud, "audit_verdict", "unknown")
                badge_color = "#10b981" if verdict in ("verified", "abstained") else "#f59e0b"
                st.markdown(
                    f'<div style="margin-top: 6px;"><span style="background-color: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}66; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; font-family: monospace;">Audit: {verdict.upper()}</span> '
                    f'<span style="color: #94a3b8; font-size: 0.8rem; font-family: monospace;">Trace: {msg.get("trace_id", "N/A")}</span></div>',
                    unsafe_allow_html=True
                )

            if msg.get("chunks"):
                with st.expander(f"🔍 Inspect {len(msg['chunks'])} Retrieved Context Chunks"):
                    for i, c in enumerate(msg["chunks"], 1):
                        score = c.get("rerank_score", c.get("score", c.get("rrf_score", 0.0)))
                        meta = c.get("metadata", {})
                        st.markdown(f"**Chunk {i}** | *Score: {score:.4f}* | `{meta.get('type')}` | *{meta.get('heading_path')}* (Page {meta.get('page')})")
                        st.code(c.get("content", ""), language="markdown")

    # Chat input
    if prompt := st.chat_input("Ask a question about the book (e.g., 'What is a qubit?', 'Compare algorithm complexities')..."):
        # Append user question
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Filters
        filters = {}
        if type_filter != "All":
            filters["type"] = type_filter
        if chapter_filter:
            filters["chapter"] = chapter_filter

        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        with st.chat_message("assistant"):
            with st.spinner("Consulting the book knowledge base..."):
                response = pipeline.query(
                    question=prompt,
                    chat_history=chat_history,
                    filters=filters if filters else None,
                    top_k=top_k,
                    strategy=strategy,
                    enable_tracing=True
                )

            st.markdown(response.answer)

            if response.sources:
                st.markdown("**Citations:**")
                source_html = " ".join([
                    f'<span class="source-tag">📖 {src.chapter} (Page {src.page})</span>'
                    for src in response.sources
                ])
                st.markdown(source_html, unsafe_allow_html=True)

            if response.audit:
                aud = response.audit
                badge_color = "#10b981" if aud.audit_verdict in ("verified", "abstained") else "#f59e0b"
                st.markdown(
                    f'<div style="margin-top: 6px;"><span style="background-color: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}66; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; font-family: monospace;">Audit: {aud.audit_verdict.upper()}</span> '
                    f'<span style="color: #94a3b8; font-size: 0.8rem; font-family: monospace;">Trace: {response.trace_id or "N/A"}</span></div>',
                    unsafe_allow_html=True
                )

            if response.retrieved_chunks:
                with st.expander(f"🔍 Inspect {len(response.retrieved_chunks)} Retrieved Context Chunks"):
                    for i, c in enumerate(response.retrieved_chunks, 1):
                        score = c.get("rerank_score", c.get("score", c.get("rrf_score", 0.0)))
                        meta = c.get("metadata", {})
                        st.markdown(f"**Chunk {i}** | *Score: {score:.4f}* | `{meta.get('type')}` | *{meta.get('heading_path')}* (Page {meta.get('page')})")
                        st.code(c.get("content", ""), language="markdown")

        # Record assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": [{"chapter": s.chapter, "page": s.page} for s in response.sources],
            "chunks": response.retrieved_chunks,
            "trace_id": response.trace_id,
            "audit": response.audit
        })
