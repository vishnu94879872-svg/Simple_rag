"""Book Tutor interactive chat view preserving core BookRAG experience with observability enhancements."""

import streamlit as st

from app.ui import empty_state, escape, page_header, pill

SUGGESTED_QUESTIONS = [
    ("Constitutional AI", "What is Constitutional AI?"),
    ("RAG Pipeline Stages", "What are the main stages of a RAG pipeline?"),
    ("Qubit definition", "What is a qubit and how is it defined?"),
    ("Quantum Superposition", "Explain quantum superposition with an example."),
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
    """Render the Chat interface with book-isolated context and 1-click report integration."""
    distinct_books = pipeline.get_distinct_books()
    book_options = ["All Books"] + distinct_books

    focus_book = st.session_state.get("selected_book_focus")
    default_idx = 0
    if focus_book and focus_book in book_options:
        default_idx = book_options.index(focus_book)

    col_hdr, col_btn = st.columns([4, 1], vertical_alignment="bottom")
    with col_hdr:
        page_header(
            "Module · Conversational QA",
            "Book Tutor",
            "Ask natural-language questions about indexed books. Every answer is grounded with chapter and page citations, then audited for grounding.",
        )

    c_book, c_strat, c_topk, c_type, c_chap = st.columns([1.6, 1.2, 0.9, 0.9, 1.2])
    with c_book:
        selected_book = st.selectbox(
            "Book focus",
            options=book_options,
            index=default_idx,
            help="Select a specific book to focus questions, or 'All Books' for multi-book search.",
        )
        st.session_state.selected_book_focus = selected_book

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

    # Per-book chat history storage
    if "chat_history_by_book" not in st.session_state:
        st.session_state.chat_history_by_book = {}

    current_history = st.session_state.chat_history_by_book.setdefault(selected_book, [])

    with col_btn:
        if st.button("Clear Conversation", type="secondary", width="stretch"):
            st.session_state.chat_history_by_book[selected_book] = []
            st.rerun()

    pending_prompt = st.session_state.pop("pending_prompt", None)

    # Dynamic suggestions based on selected book or all available books
    active_suggestions = []
    if selected_book == "Sample Quantum" or (selected_book == "All Books" and "Sample Quantum" in distinct_books):
        active_suggestions.append(("Qubit definition", "What is a qubit and how is it defined?"))
        active_suggestions.append(("Quantum Superposition", "Explain quantum superposition with an example."))
    if "Animal" in selected_book or (selected_book == "All Books" and any("Animal" in b for b in distinct_books)):
        active_suggestions.append(("Animal Feeding", "How do different mammals feed and digest their food?"))
        active_suggestions.append(("Bird Features", "What features distinguish birds from other vertebrates?"))
    if "Anthropic" in selected_book or (selected_book == "All Books" and any("Anthropic" in b for b in distinct_books)):
        active_suggestions.append(("Constitutional AI", "What is Constitutional AI?"))
        active_suggestions.append(("RAG Pipeline Stages", "What are the main stages of a RAG pipeline?"))

    if not active_suggestions:
        active_suggestions = SUGGESTED_QUESTIONS

    if not current_history and pending_prompt is None:
        st.markdown('<div class="br-nav-label">Suggested questions</div>', unsafe_allow_html=True)
        display_suggestions = active_suggestions[:4]
        suggestion_cols = st.columns(len(display_suggestions))
        for index, (col, (label, question)) in enumerate(zip(suggestion_cols, display_suggestions)):
            with col:
                if st.button(label, key=f"suggested_q_{index}", width="stretch"):
                    st.session_state["pending_prompt"] = question

        pending_prompt = st.session_state.pop("pending_prompt", None)
        if pending_prompt is None:
            focus_msg = f"currently tutoring on '{selected_book}'" if selected_book != "All Books" else "searching across all indexed books"
            empty_state(
                "Grounded answers, cited to the exact page",
                f"Pick a suggestion above or ask your own question ({focus_msg}). Every response carries an audit verdict and page-specific citations.",
                mark="📖",
            )

    # Render previous messages for the active book
    for idx, msg in enumerate(current_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])
            if msg.get("audit"):
                st.markdown(_audit_badge(msg["audit"], msg.get("trace_id")), unsafe_allow_html=True)
            if msg.get("chunks"):
                _render_chunks(msg["chunks"])

            if msg["role"] == "assistant":
                c_rpt, _ = st.columns([1.2, 3.8])
                with c_rpt:
                    if st.button("🚩 Report Issue with Answer", key=f"rpt_{idx}", type="secondary"):
                        st.session_state.prefill_ticket = {
                            "book_title": selected_book if selected_book != "All Books" else None,
                            "query": msg.get("query", ""),
                            "category": "Incorrect Answer / Hallucination",
                            "description": f"Issue reported on answer:\n\nQuestion: '{msg.get('query', '')}'\nResponse: {msg.get('content', '')[:250]}...\n\nTrace ID: {msg.get('trace_id', 'N/A')}",
                        }
                        st.session_state.user_nav = "Support & Reports"
                        st.rerun()

    typed_prompt = st.chat_input(f"Ask a question about {selected_book}..." if selected_book != "All Books" else "Ask a question about indexed books...")
    prompt = typed_prompt or pending_prompt
    if not prompt:
        return

    current_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    filters = {}
    if selected_book != "All Books":
        filters["book_title"] = selected_book
    if type_filter != "All":
        filters["type"] = type_filter
    if chapter_filter:
        filters["chapter"] = chapter_filter

    chat_history = [{"role": m["role"], "content": m["content"]} for m in current_history[:-1]]

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

    current_history.append(
        {
            "role": "assistant",
            "query": prompt,
            "content": response.answer,
            "sources": [{"chapter": s.chapter, "page": s.page} for s in response.sources],
            "chunks": response.retrieved_chunks,
            "trace_id": response.trace_id,
            "audit": response.audit,
        }
    )
    st.rerun()
