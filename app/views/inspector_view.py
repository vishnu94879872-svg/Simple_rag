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
from app.ui import empty_state, escape, page_header, pill


def render_chunk_inspector():
    """Render Module C - PDF and Chunk Inspector."""
    page_header(
        "Module · Inspection",
        "PDF & Chunk Inspector",
        "Compare original document pages with extracted text, generated chunks, and heuristic quality findings side by side.",
    )

    book_files = list(Path(settings.data_dir / "books").glob("*.pdf")) if (settings.data_dir / "books").exists() else []
    if not book_files:
        empty_state(
            "No PDF books found",
            "Add a PDF to data/books/ or ingest one from the Ingestion Monitor, then return to inspect its pages and chunks.",
            mark="▤",
        )
        return

    # Select document
    col_sel, col_thresh = st.columns([3, 1], gap="medium", vertical_alignment="bottom")
    with col_sel:
        selected_pdf_name = st.selectbox("Document to inspect", options=[b.name for b in book_files], index=0, width="stretch")
        pdf_path = settings.data_dir / "books" / selected_pdf_name

    with col_thresh:
        with st.popover("Quality thresholds", width="stretch"):
            min_chars = st.slider("Min chunk characters", min_value=20, max_value=200, value=80, step=10)
            max_chars = st.slider("Max chunk characters", min_value=500, max_value=2500, value=1200, step=50)
            max_overlap = st.slider("Max overlap threshold", min_value=0.1, max_value=0.8, value=0.40, step=0.05)
            st.caption("Diagnostic heuristics only. Changing thresholds does not alter stored vectors.")

    # Open PDF
    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)

    # Page selector
    selected_page_num = st.slider("Document page", min_value=1, max_value=total_pages, value=1, step=1)
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
        st.markdown(f"### Original PDF · page {selected_page_num} of {total_pages}")
        try:
            # Render page to high-res image
            pix = page.get_pixmap(dpi=130)
            img_bytes = pix.tobytes("png")
            st.image(img_bytes, caption=f"{selected_pdf_name} (Page {selected_page_num})", width="stretch")
        except Exception as e:
            st.error(f"Could not render PDF raster: {e}")
            st.text_area("Raw Text Fallback", value=page.get_text(), height=450)

    # --- RIGHT PANEL: Extracted Text & Chunks ---
    with right_panel:
        st.markdown("### Extracted text & chunk diagnostics")

        # Tab view: Page text vs Chunks
        tab_chunks, tab_raw_text = st.tabs([f"Chunks ({len(page_chunks)})", "Extracted page text"])

        with tab_raw_text:
            extracted_raw = page.get_text()
            st.caption(f"Raw character length: {len(extracted_raw):,} · word count: {len(extracted_raw.split()):,}")
            st.text_area(
                "Extracted text (wrapped)",
                value=extracted_raw,
                height=450,
                help="Text is wrapped to prevent horizontal overflow.",
            )

        # Top Diagnostics & Config Banner
        with st.expander("Document ingestion & chunking configuration", expanded=False):
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
        with st.expander("Test query retrieval rank on this page", expanded=False):
            rank_query = st.text_input("Query to score", value="quantum superposition", key=f"rank_q_{selected_page_num}")
            if rank_query.strip():
                from retrieval.bm25_store import tokenize
                q_tokens = set(tokenize(rank_query))
                st.caption(f"Scoring {len(page_chunks)} chunks on page {selected_page_num} against query tokens: `{', '.join(q_tokens)}`")

        with tab_chunks:
            if not page_chunks:
                empty_state(
                    "No chunks on this page",
                    "This page may contain only images or blank margins. Try another page or inspect the extracted text tab.",
                    mark="—",
                )
            else:
                chunk_index_map = {c.chunk_id: i for i, c in enumerate(all_chunks)}

                for idx, c in enumerate(page_chunks, 1):
                    rep = report_by_id.get(c.chunk_id)
                    meta = c.metadata
                    global_idx = chunk_index_map.get(c.chunk_id, 0)
                    prev_chunk = all_chunks[global_idx - 1] if global_idx > 0 else None
                    next_chunk = all_chunks[global_idx + 1] if global_idx < len(all_chunks) - 1 else None

                    # Calculate query token match score if query entered
                    match_block = ""
                    if "rank_query" in locals() and rank_query.strip():
                        c_toks = set(tokenize(c.content))
                        matched_toks = q_tokens.intersection(c_toks)
                        if matched_toks:
                            terms = ", ".join(sorted(escape(token) for token in matched_toks))
                            match_block = (
                                f'<div class="chunk-match">'
                                f'{pill(f"Query match {len(matched_toks)}/{len(q_tokens)} · {terms}", "info")}'
                                f"</div>"
                            )

                    issues_html = ""
                    if rep and rep.issues:
                        issue_blocks = []
                        for issue in rep.issues:
                            is_confirmed_error = issue.severity == "error"
                            tone = "danger" if is_confirmed_error else ("warning" if issue.severity == "warning" else "info")
                            label_tag = "Confirmed error" if is_confirmed_error else ("Heuristic warning" if issue.severity == "warning" else "Info")
                            plain_class = " chunk-issue--plain" if tone == "info" else ""
                            evidence_html = (
                                f'<div class="chunk-evidence">EVIDENCE · {escape(issue.evidence)}</div>'
                                if issue.evidence
                                else ""
                            )
                            issue_blocks.append(
                                f'<div class="br-verdict br-verdict--{tone} chunk-issue{plain_class}">'
                                f'<div class="br-verdict-title">{escape(label_tag)} · {escape(issue.issue_type.replace("_", " ").title())}</div>'
                                f'<div class="br-verdict-body">{escape(issue.description)}</div>'
                                f"{evidence_html}"
                                f'<div class="chunk-recommendation">Recommendation · {escape(issue.recommendation)}</div>'
                                f"</div>"
                            )
                        issues_html = "".join(issue_blocks)

                    retrieval_html = ""
                    if rep:
                        ret_color = "var(--br-success)" if rep.was_retrieved else "var(--br-border-strong)"
                        queries_html = (
                            f'<div class="chunk-queries">Queries · {", ".join(escape(q) for q in rep.queries_retrieved_by[:2])}</div>'
                            if rep.queries_retrieved_by
                            else ""
                        )
                        best_rank_html = pill(f"Best rank #{rep.best_rank}", "success") if rep.best_rank else ""
                        retrieval_html = (
                            f'<div class="br-stage chunk-retrieval" style="border-left-color:{ret_color};">'
                            f'<div class="chunk-retrieval-row">'
                            f'<span class="chunk-retrieval-status">'
                            f'{"Retrieved in recorded queries" if rep.was_retrieved else "Not yet retrieved in recorded traces"}</span>'
                            f"{best_rank_html}"
                            f"</div>"
                            f"{queries_html}"
                            f'<div class="chunk-causality">Causality · {escape(rep.causality_assessment or "Nominal")}</div>'
                            f"</div>"
                        )

                    with st.container():
                        st.markdown(
                            f'<div class="chunk-card">'
                            f'<div class="chunk-head">'
                            f'<span class="chunk-id">CHUNK #{idx} · {escape(c.chunk_id)}</span>'
                            f'<span class="chunk-tags">{pill(meta.type, "accent")}</span>'
                            f"</div>"
                            f'<div class="chunk-meta">Page {escape(meta.page)} · {escape(meta.heading_path)} · ~{rep.token_count_estimate if rep else "?"} tokens · {len(c.content):,} chars</div>'
                            f"{match_block}"
                            f"{issues_html}"
                            f"{retrieval_html}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        with st.expander("View chunk content & neighboring boundaries"):
                            st.markdown("**Chunk content (preserves code, tables, and whitespace)**")
                            st.code(c.content, language="markdown")
                            st.markdown(f"**Overlap with previous chunk:** {rep.overlap_prev_chars} chars ({int(rep.overlap_prev_pct * 100)}%)" if rep else "")
                            st.markdown(f"**Parent section ID:** `{meta.parent_id}` · **Keywords:** `{meta.keywords}`")

                            c_prev, c_next = st.columns(2)
                            with c_prev:
                                st.markdown("**Preceding neighbor**")
                                if prev_chunk:
                                    st.caption(f"ID: `{prev_chunk.chunk_id}` (Page {prev_chunk.metadata.page})")
                                    st.text(prev_chunk.content[:150] + "..." if len(prev_chunk.content) > 150 else prev_chunk.content)
                                else:
                                    st.caption("None (first chunk in document)")
                            with c_next:
                                st.markdown("**Succeeding neighbor**")
                                if next_chunk:
                                    st.caption(f"ID: `{next_chunk.chunk_id}` (Page {next_chunk.metadata.page})")
                                    st.text(next_chunk.content[:150] + "..." if len(next_chunk.content) > 150 else next_chunk.content)
                                else:
                                    st.caption("None (last chunk in document)")

