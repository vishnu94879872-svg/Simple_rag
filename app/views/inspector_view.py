"""Module C: Two-panel PDF and Chunk Inspector with Quality Checker."""

import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional
import pymupdf

from config import settings
from ingest.pdf_parser import PDFParser
from ingest.text_splitter import ChunkBuilder
from observability.chunk_inspector import ChunkQualityChecker
from ingest.models import Chunk, ContentType


def render_chunk_inspector():
    """Render Module C - PDF and Chunk Inspector."""
    st.markdown("## 🔍 PDF & Chunk Inspector")
    st.caption("Side-by-side verification comparing original document pages with extracted text, generated chunks, and quality heuristics.")

    book_files = list(Path(settings.data_dir / "books").glob("*.pdf")) if (settings.data_dir / "books").exists() else []
    if not book_files:
        st.warning("No PDF books found in `data/books/`. Ingest a book first to inspect its pages and chunks.")
        return

    # Select document
    col_sel, col_thresh = st.columns([2, 1])
    with col_sel:
        selected_pdf_name = st.selectbox("Select Document to Inspect:", options=[b.name for b in book_files], index=0)
        pdf_path = settings.data_dir / "books" / selected_pdf_name

    with col_thresh:
        with st.popover("⚙️ Quality Thresholds"):
            min_chars = st.slider("Min Chunk Characters", min_value=20, max_value=200, value=80, step=10)
            max_chars = st.slider("Max Chunk Characters", min_value=500, max_value=2500, value=1200, step=50)
            max_overlap = st.slider("Max Overlap Threshold", min_value=0.1, max_value=0.8, value=0.40, step=0.05)
            st.caption("Diagnostic heuristics only. Changing thresholds does NOT alter stored vectors.")

    # Open PDF
    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)

    # Page selector
    selected_page_num = st.slider("Select Document Page:", min_value=1, max_value=total_pages, value=1, step=1)
    page_idx = selected_page_num - 1
    page = doc[page_idx]

    # Parse chunks for document
    @st.cache_data(show_spinner="Extracting document chunks for inspector...")
    def get_parsed_data(file_str: str):
        p = PDFParser(Path(file_str))
        blocks = p.parse()
        builder = ChunkBuilder(book_title=p.book_title)
        chunks, parents = builder.build_chunks_and_parents(blocks)
        return blocks, chunks

    blocks, all_chunks = get_parsed_data(str(pdf_path))

    # Filter chunks for current page
    page_chunks = [c for c in all_chunks if c.metadata.page == selected_page_num]

    # Evaluate chunk quality reports
    quality_checker = ChunkQualityChecker(
        min_chars=min_chars,
        max_chars=max_chars,
        max_overlap_pct=max_overlap
    )
    quality_reports = quality_checker.evaluate_chunks(page_chunks)
    report_by_id = {r.chunk_id: r for r in quality_reports}

    # Two-panel layout
    left_panel, right_panel = st.columns([1, 1.2], gap="medium")

    # --- LEFT PANEL: Original PDF View ---
    with left_panel:
        st.markdown(f"### 📄 Original PDF — Page {selected_page_num} of {total_pages}")
        try:
            # Render page to high-res image
            pix = page.get_pixmap(dpi=130)
            img_bytes = pix.tobytes("png")
            st.image(img_bytes, caption=f"{selected_pdf_name} (Page {selected_page_num})", use_container_width=True)
        except Exception as e:
            st.error(f"Could not render PDF raster: {e}")
            st.text_area("Raw Text Fallback", value=page.get_text(), height=450)

    # --- RIGHT PANEL: Extracted Text & Chunks ---
    with right_panel:
        st.markdown(f"### 🧬 Extracted Text & Chunk Diagnostics")

        # Tab view: Page text vs Chunks
        tab_chunks, tab_raw_text = st.tabs([f"📦 Chunks ({len(page_chunks)})", "📝 Extracted Page Text"])

        with tab_raw_text:
            extracted_raw = page.get_text()
            st.caption(f"Raw character length: {len(extracted_raw)} | Word count: {len(extracted_raw.split())}")
            # Text area with wrap to avoid horizontal overflow
            st.text_area(
                "Extracted Text (Wrapped)",
                value=extracted_raw,
                height=450,
                help="Fixed horizontal overflow with automatic word wrapping"
            )

        # Top Diagnostics & Config Banner
        with st.expander("⚙️ Document Ingestion & Chunking Configuration", expanded=False):
            c_cfg1, c_cfg2, c_cfg3 = st.columns(3)
            with c_cfg1:
                st.markdown("**Chunking Strategy:** `RecursiveHeadingAware`")
                st.markdown("**Target Chunk Size:** `1000 characters`")
            with c_cfg2:
                st.markdown("**Chunk Overlap:** `150 characters`")
                st.markdown("**Embedding Model:** `BAAI/bge-small-en-v1.5`")
            with c_cfg3:
                st.markdown(f"**Document ID:** `{selected_pdf_name}`")
                st.markdown(f"**Total Chunks in Book:** `{len(all_chunks)}`")

        # Quick Retrieval Rank Checker for this page's chunks
        with st.expander("🔍 Test Query Retrieval Rank on this Page", expanded=False):
            rank_query = st.text_input("Enter Query to Score:", value="quantum superposition", key=f"rank_q_{selected_page_num}")
            if rank_query.strip():
                from retrieval.bm25_store import tokenize
                q_tokens = set(tokenize(rank_query))
                st.caption(f"Scoring {len(page_chunks)} chunks on page {selected_page_num} against query tokens: `{', '.join(q_tokens)}`")

        with tab_chunks:
            if not page_chunks:
                st.info(f"No discrete text chunks generated for page {selected_page_num}. Page may contain exclusively images or blank margins.")
            else:
                chunk_index_map = {c.chunk_id: i for i, c in enumerate(all_chunks)}

                for idx, c in enumerate(page_chunks, 1):
                    rep = report_by_id.get(c.chunk_id)
                    meta = c.metadata
                    global_idx = chunk_index_map.get(c.chunk_id, 0)
                    prev_chunk = all_chunks[global_idx - 1] if global_idx > 0 else None
                    next_chunk = all_chunks[global_idx + 1] if global_idx < len(all_chunks) - 1 else None

                    # Calculate query token match score if query entered
                    term_match_score = ""
                    if 'rank_query' in locals() and rank_query.strip():
                        c_toks = set(tokenize(c.content))
                        matched_toks = q_tokens.intersection(c_toks)
                        if matched_toks:
                            term_match_score = f" | 🎯 <b>Query Match:</b> {len(matched_toks)}/{len(q_tokens)} terms ({', '.join(matched_toks)})"

                    # Card container
                    with st.container():
                        st.markdown(
                            f'<div class="chunk-card" style="word-break: break-word; overflow-x: auto;">'
                            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
                            f'<span style="font-weight: 700; color: #60a5fa; font-family: monospace;">Chunk #{idx}: {c.chunk_id}</span>'
                            f'<span style="background-color: #27272a; color: #94a3b8; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">Doc: {selected_pdf_name} | Type: {meta.type}</span>'
                            f'</div>'
                            f'<div style="color: #94a3b8; font-size: 0.82rem; margin-bottom: 8px;">'
                            f'<b>Page:</b> {meta.page} | <b>Section:</b> {meta.heading_path} | <b>Tokens:</b> ~{rep.token_count_estimate if rep else "?"} | <b>Chars:</b> {len(c.content)}{term_match_score}'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        # Quality Issue Badges with Exact Evidence
                        if rep and rep.issues:
                            for issue in rep.issues:
                                is_confirmed_error = issue.severity == "error"
                                badge_col = "#ef4444" if is_confirmed_error else "#f59e0b"
                                label_tag = "CONFIRMED ERROR" if is_confirmed_error else "HEURISTIC WARNING"
                                evidence_html = f'<div style="color: #38bdf8; font-size: 0.78rem; margin-top: 2px;">🔎 <b>Evidence:</b> {issue.evidence}</div>' if issue.evidence else ""
                                st.markdown(
                                    f'<div style="background-color: #18181b; border-left: 3px solid {badge_col}; padding: 6px 10px; margin-bottom: 6px; font-size: 0.82rem;">'
                                    f'<b style="color: {badge_col};">[{label_tag}: {issue.issue_type.upper()}]</b> {issue.description}'
                                    f'{evidence_html}'
                                    f'<div style="color: #a8a29e; font-size: 0.78rem; margin-top: 2px;">💡 {issue.recommendation}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                        # Retrieval-Aware Chunk Diagnostics
                        if rep:
                            ret_color = "#10b981" if rep.was_retrieved else "#64748b"
                            queries_str = ", ".join([f"<i>{q}</i>" for q in rep.queries_retrieved_by[:2]])
                            queries_html = f'<div style="color: #94a3b8; font-size: 0.78rem; margin-top: 3px;"><b>Queries:</b> {queries_str}</div>' if rep.queries_retrieved_by else ""
                            st.markdown(
                                f'<div style="background-color: #0f172a; border: 1px solid {ret_color}44; border-radius: 6px; padding: 8px 12px; margin: 6px 0; font-size: 0.82rem;">'
                                f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                                f'<span><b>🎯 Retrieval Status:</b> {"Retrieved in Recorded Queries" if rep.was_retrieved else "Not Yet Retrieved in Recorded Traces"}</span>'
                                f'<span>{"🏆 Best Rank: #" + str(rep.best_rank) if rep.best_rank else ""}</span>'
                                f'</div>'
                                f'{queries_html}'
                                f'<div style="color: #38bdf8; font-size: 0.78rem; margin-top: 3px;"><b>Causality Assessment:</b> {rep.causality_assessment or "Nominal"}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                        # Expandable Chunk Content & Neighboring Chunks
                        with st.expander("📄 View Chunk Content & Neighboring Boundaries"):
                            st.markdown("**Chunk Content (Preserves Code, Tables, & Whitespace):**")
                            st.code(c.content, language="markdown")
                            st.markdown(f"**Overlap with Previous Chunk:** {rep.overlap_prev_chars} chars ({int(rep.overlap_prev_pct * 100)}%)" if rep else "")
                            st.markdown(f"**Parent Section ID:** `{meta.parent_id}` | **Keywords:** `{meta.keywords}`")

                            c_prev, c_next = st.columns(2)
                            with c_prev:
                                st.markdown("**⏮️ Preceding Neighbor:**")
                                if prev_chunk:
                                    st.caption(f"ID: `{prev_chunk.chunk_id}` (Page {prev_chunk.metadata.page})")
                                    st.text(prev_chunk.content[:150] + "..." if len(prev_chunk.content) > 150 else prev_chunk.content)
                                else:
                                    st.caption("None (First chunk in document)")
                            with c_next:
                                st.markdown("**⏭️ Succeeding Neighbor:**")
                                if next_chunk:
                                    st.caption(f"ID: `{next_chunk.chunk_id}` (Page {next_chunk.metadata.page})")
                                    st.text(next_chunk.content[:150] + "..." if len(next_chunk.content) > 150 else next_chunk.content)
                                else:
                                    st.caption("None (Last chunk in document)")

                        st.markdown("<hr style='border: 0; border-top: 1px solid #27272a; margin: 10px 0;'>", unsafe_allow_html=True)

