"""Book Tutor interactive chat view preserving core BookRAG experience with observability enhancements."""

import streamlit as st

from app.ui import empty_state, escape, page_header, pill

SUGGESTED_QUESTIONS = [
    ("Qubit definition", "What is a qubit and how is it defined?"),
    ("Superposition example", "Explain quantum superposition with an example."),
    ("Shor vs RSA", "How does Shor's algorithm challenge RSA encryption?"),
]


def _audit_badge(audit, trace_id) -> str:
    verdict = getattr(audit, "audit_verdict", "unknown")
    tone = "success" if verdict in ("verified", "abstained") else ("warning" if verdict == "partially_grounded" else "danger")
    trace_html = (
        f'<span style="color:var(--br-text-dim);font-family:var(--br-mono);font-size:10px;margin-left:8px;">'
        f"TRACE {escape(trace_id or 'N/A')}</span>"
    )
    return f'<div style="margin-top:8px;">{pill(f"Audit · {verdict}", tone)}{trace_html}</div>'


def _render_sources(sources) -> None:
    source_html = " ".join(
        f'<span class="source-tag">{escape(src["chapter"])} · Page {escape(src["page"])}</span>' for src in sources
    )
    st.markdown(source_html, unsafe_allow_html=True)


def _render_chunks(chunks) -> None:
    with st.expander(f"Inspect {len(chunks)} retrieved context chunks"):
        for i, chunk in enumerate(chunks, 1):
            score = chunk.get("rerank_score", chunk.get("score", chunk.get("rrf_score", 0.0)))
            meta = chunk.get("metadata", {})
            st.markdown(
                f"**Chunk {i}** · score `{score:.4f}` · type `{meta.get('type')}` · "
                f"{meta.get('heading_path')} (Page {meta.get('page')})"
            )
            st.code(chunk.get("content", ""), language="markdown")


def render_chat(pipeline):
    """Render the Book Tutor chat interface."""
    page_header(
        "Module · Conversational QA",
        "Book Tutor",
        "Ask natural-language questions about indexed books. Every answer is grounded with chapter and page citations, then audited for grounding.",
    )

    c_strat, c_topk, c_type, c_chap = st.columns([1.5, 1, 1, 1.5])
    with c_strat:
        strategy = st.selectbox(
            "Retrieval strategy",
            options=["reranked", "hybrid_rrf", "vector_only", "bm25_only"],
            index=0,
            help="Select the retrieval pipeline variant.",
        )
    with c_topk:
        top_k = st.selectbox("Top-k chunks", options=[3, 5, 10], index=1)
    with c_type:
        type_filter = st.selectbox("Content type", options=["All", "text", "table", "image", "code", "list"], index=0)
    with c_chap:
        chapter_filter = st.text_input("Chapter filter (optional)", value="").strip()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    pending_prompt = st.session_state.pop("pending_prompt", None)

    if not st.session_state.messages and pending_prompt is None:
        st.markdown('<div class="br-nav-label">Suggested questions</div>', unsafe_allow_html=True)
        suggestion_cols = st.columns(len(SUGGESTED_QUESTIONS))
        for index, (col, (label, question)) in enumerate(zip(suggestion_cols, SUGGESTED_QUESTIONS)):
            with col:
                if st.button(label, key=f"suggested_q_{index}", width="stretch"):
                    st.session_state["pending_prompt"] = question

        pending_prompt = st.session_state.pop("pending_prompt", None)
        if pending_prompt is None:
            empty_state(
                "Grounded answers, cited to the page",
                "Pick a suggestion above or ask your own question. Every response carries an audit verdict and a trace ID you can replay in the Trace Viewer.",
                mark="◇",
            )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])
            if msg.get("audit"):
                st.markdown(_audit_badge(msg["audit"], msg.get("trace_id")), unsafe_allow_html=True)
            if msg.get("chunks"):
                _render_chunks(msg["chunks"])

    typed_prompt = st.chat_input("Ask a question about the book...")
    prompt = typed_prompt or pending_prompt
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    filters = {}
    if type_filter != "All":
        filters["type"] = type_filter
    if chapter_filter:
        filters["chapter"] = chapter_filter

    chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        with st.spinner("Consulting the book knowledge base..."):
            response = pipeline.query(
                question=prompt,
                chat_history=chat_history,
                filters=filters if filters else None,
                top_k=top_k,
                strategy=strategy,
                enable_tracing=True,
            )

        st.markdown(response.answer)

        if response.sources:
            _render_sources([{"chapter": s.chapter, "page": s.page} for s in response.sources])
        if response.audit:
            st.markdown(_audit_badge(response.audit, response.trace_id), unsafe_allow_html=True)
        if response.retrieved_chunks:
            _render_chunks(response.retrieved_chunks)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": [{"chapter": s.chapter, "page": s.page} for s in response.sources],
            "chunks": response.retrieved_chunks,
            "trace_id": response.trace_id,
            "audit": response.audit,
        }
    )
    st.rerun()
