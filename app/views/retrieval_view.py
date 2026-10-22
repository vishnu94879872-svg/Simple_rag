"""Module D: Retrieval Playground view for query debugging and strategy comparison."""

import streamlit as st
from typing import List, Dict, Any

from observability.retrieval_debugger import retrieval_debugger
from observability.models import RetrievalStrategy
from app.ui import empty_state, escape, page_header, pill


def render_retrieval_playground():
    """Render Module D - Retrieval Playground."""
    page_header(
        "Module · Retrieval",
        "Retrieval Playground",
        "Inspect and compare dense vector search, BM25, hybrid RRF, and neural reranking candidates before generation.",
    )

    # Query Input
    query = st.text_input(
        "Query to inspect",
        value="What is a qubit and how is it defined?",
        help="Input query to debug retrieval without generating an LLM response."
    )

    tab_single, tab_compare = st.tabs(["Single strategy debugger", "Side-by-side comparator"])

    # --- TAB 1: Single Strategy ---
    with tab_single:
        c1, c2, c3 = st.columns([1.5, 1, 1.5])
        with c1:
            strategy = st.selectbox(
                "Strategy",
                options=[
                    RetrievalStrategy.RERANKED.value,
                    RetrievalStrategy.HYBRID_RRF.value,
                    RetrievalStrategy.VECTOR_ONLY.value,
                    RetrievalStrategy.BM25_ONLY.value,
                ],
                index=0,
                key="single_strat"
            )
        with c2:
            top_k = st.selectbox("Top-k Setting", options=[3, 5, 10], index=1, key="single_topk")
        with c3:
            content_type = st.selectbox(
                "Metadata Type Filter",
                options=["All", "text", "table", "image", "code", "list"],
                index=0,
                key="single_type"
            )

        _, c_run = st.columns([2, 1])
        with c_run:
            run_single = st.button("Execute retrieval", key="btn_run_single", type="primary", width="stretch")

        if run_single:
            filters = {"type": content_type} if content_type != "All" else None

            with st.spinner("Executing retrieval strategy..."):
                res = retrieval_debugger.run_query(
                    query=query,
                    strategy=strategy,
                    top_k=top_k,
                    filters=filters
                )

            # Execution metrics
            st.markdown("### Retrieval diagnostics")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Latency", f"{res['duration_ms']:.1f} ms")
            with m2:
                st.metric("Dense Candidates", res["vector_candidates_count"])
            with m3:
                st.metric("BM25 Candidates", res["bm25_candidates_count"])
            with m4:
                st.metric("Final Hits Returned", len(res["candidates"]))

            st.markdown(f"**Effective Query (Rewritten):** `{res['rewritten_query']}`")

            # Candidate List
            st.markdown("### Retrieved candidates")
            if not res["candidates"]:
                empty_state(
                    "Zero candidates retrieved",
                    "The query terms did not match the dense or sparse index. Try a broader query or a different strategy.",
                    mark="∅",
                )
            else:
                for c in res["candidates"]:
                    meta = c.metadata
                    score_label = c.score_type.replace("_", " ").title()
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">'
                        f'<span style="font-weight:600;color:var(--br-accent-strong);font-family:var(--br-mono);font-size:12px;">RANK #{c.rank} · {escape(c.chunk_id)}</span>'
                        f'{pill(f"{score_label} · {c.score:.4f}", "info")}'
                        f'</div>'
                        f'<div style="color:var(--br-text-muted);font-size:12px;margin-top:7px;line-height:1.55;">'
                        f'{escape(meta.get("chapter", "Unknown chapter"))} · page {escape(meta.get("page", "—"))} · '
                        f'<span style="font-family:var(--br-mono);font-size:10.5px;">{escape(meta.get("type", "text"))}</span> · '
                        f'{escape(meta.get("heading_path", ""))}'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    with st.expander(f"View chunk text ({len(c.content):,} chars)"):
                        st.code(c.content, language="markdown")

    # --- TAB 2: Comparison Mode ---
    with tab_compare:
        st.caption("Compare how changing retrieval strategies or top-k settings changes candidate rankings and content coverage.")
        col_a, col_b = st.columns(2)
        with col_a:
            strat_a = st.selectbox("Baseline Strategy (A):", options=["vector_only", "bm25_only", "hybrid_rrf", "reranked"], index=0, key="cmp_a")
        with col_b:
            strat_b = st.selectbox("Comparison Strategy (B):", options=["vector_only", "bm25_only", "hybrid_rrf", "reranked"], index=3, key="cmp_b")

        cmp_top_k = st.selectbox("Top-k for Comparison:", options=[3, 5, 10], index=1, key="cmp_topk")

        _, c_compare = st.columns([2, 1])
        with c_compare:
            run_compare = st.button("Compare strategies", type="primary", width="stretch")

        if run_compare:
            with st.spinner("Benchmarking both retrieval strategies..."):
                comparison = retrieval_debugger.compare_strategies(
                    query=query,
                    strategy_a=strat_a,
                    strategy_b=strat_b,
                    top_k=cmp_top_k
                )

            # Metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Common Chunks", f"{len(comparison.common_chunk_ids)} / {cmp_top_k}")
            with c2:
                st.metric(f"Only in Strategy A ({strat_a})", len(comparison.only_in_a_chunk_ids))
            with c3:
                st.metric(f"Only in Strategy B ({strat_b})", len(comparison.only_in_b_chunk_ids))

            col_res_a, col_res_b = st.columns(2)
            with col_res_a:
                st.markdown(f"#### Strategy A · `{strat_a}` · {comparison.duration_a_ms:.1f} ms")
                for c in comparison.hits_a:
                    is_common = c.chunk_id in comparison.common_chunk_ids
                    tag = pill("Common", "success") if is_common else pill("Only in A", "warning")
                    st.markdown(
                        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:10px;'
                        f'padding:7px 0;border-bottom:1px solid var(--br-border);">'
                        f'<span style="font-size:12px;color:var(--br-text-muted);">Rank {c.rank} · <span style="font-family:var(--br-mono);">{escape(c.chunk_id)}</span></span>'
                        f'<span style="display:flex;align-items:center;gap:8px;">{tag}'
                        f'<span style="font-family:var(--br-mono);font-size:10.5px;color:var(--br-text-dim);">{c.score:.4f}</span></span></div>',
                        unsafe_allow_html=True,
                    )

            with col_res_b:
                st.markdown(f"#### Strategy B · `{strat_b}` · {comparison.duration_b_ms:.1f} ms")
                for c in comparison.hits_b:
                    is_common = c.chunk_id in comparison.common_chunk_ids
                    tag = pill("Common", "success") if is_common else pill("Only in B", "info")
                    st.markdown(
                        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:10px;'
                        f'padding:7px 0;border-bottom:1px solid var(--br-border);">'
                        f'<span style="font-size:12px;color:var(--br-text-muted);">Rank {c.rank} · <span style="font-family:var(--br-mono);">{escape(c.chunk_id)}</span></span>'
                        f'<span style="display:flex;align-items:center;gap:8px;">{tag}'
                        f'<span style="font-family:var(--br-mono);font-size:10.5px;color:var(--br-text-dim);">{c.score:.4f}</span></span></div>',
                        unsafe_allow_html=True,
                    )
