"""Module B: Document Ingestion Diagnostics view."""

import streamlit as st
from pathlib import Path
import time

from config import settings
from observability.storage import storage
from observability.ingestion_monitor import ingestion_monitor
from observability.models import IngestionJob, StageStatus, IngestionStageInfo


def render_ingestion_monitor():
    """Render Module B - Document Ingestion Diagnostics."""
    st.markdown("## 📥 Document Ingestion Diagnostics")
    st.caption("Deep inspection of PDF parsing, structural segmentation, chunk generation, and index synchronization.")

    # Top Ingestion Action / Upload
    with st.expander("📤 Ingest & Monitor New PDF Document", expanded=False):
        uploaded_file = st.file_uploader("Upload PDF Book to Ingest & Benchmark", type=["pdf"])
        if uploaded_file is not None:
            if st.button("🚀 Run Diagnostic Ingestion", use_container_width=True):
                save_path = settings.data_dir / "books" / uploaded_file.name
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.status(f"Ingesting '{uploaded_file.name}' through monitored pipeline...", expanded=True) as status:
                    st.write("🔍 Running structural pre-flight inspection...")
                    job = ingestion_monitor.run_monitored_ingestion(save_path)
                    if job.overall_status == StageStatus.COMPLETED:
                        status.update(label=f"✅ Successfully indexed '{uploaded_file.name}'!", state="complete")
                    elif job.overall_status == StageStatus.WARNING:
                        status.update(label=f"⚠️ Ingested '{uploaded_file.name}' with heuristic warnings.", state="complete")
                    else:
                        status.update(label=f"❌ Ingestion failed for '{uploaded_file.name}'.", state="error")
                st.rerun()

    # List recorded jobs
    jobs = storage.list_jobs(limit=50)

    # Re-index existing files if no jobs yet
    book_files = list(Path(settings.data_dir / "books").glob("*.pdf")) if (settings.data_dir / "books").exists() else []

    if not jobs and book_files:
        st.info("Found existing PDF books in `data/books/` that have not yet been evaluated by the Ingestion Monitor.")
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            selected_pdf = st.selectbox("Select document to diagnose:", options=[b.name for b in book_files])
            if st.button(f"Run Diagnostics on '{selected_pdf}'"):
                target = settings.data_dir / "books" / selected_pdf
                with st.spinner("Analyzing document structure..."):
                    job = ingestion_monitor.run_monitored_ingestion(target)
                st.success(f"Diagnostics complete for {selected_pdf}!")
                st.rerun()

    if not jobs:
        st.warning("No ingestion jobs recorded yet. Upload a PDF above to begin diagnostic monitoring.")
        return

    # Job selection
    job_options = {f"{j.filename} ({j.job_id}) - {j.overall_status.upper()}": j for j in jobs}
    selected_label = st.selectbox("Select Ingestion Job to Inspect:", options=list(job_options.keys()))
    job: IngestionJob = job_options[selected_label]

    # --- Job Header Summary ---
    st.markdown("### 📋 Document Metadata & Execution Summary")
    
    struct_details = job.stages.get("structural_inspection", IngestionStageInfo(stage_name="")).details or {}
    phys_pages = struct_details.get("physical_pages", job.page_count)
    extracted_pages = struct_details.get("successfully_extracted_pages", job.page_count)
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Document ID", job.document_id[:16])
    with m2:
        st.metric("File Size", f"{job.file_size_bytes / 1024:.1f} KB")
    with m3:
        st.metric("Physical Pages", phys_pages, help="Total physical page count in source PDF file")
    with m4:
        st.metric("Extracted Pages", extracted_pages, help="Pages successfully yielding readable text, tables, or drawings")
    with m5:
        st.metric("Indexed Chunks", job.chunk_count, help="Searchable retrieval chunks after recursive sentence-safe splitting")
    with m6:
        status_color = "#10b981" if job.overall_status == StageStatus.COMPLETED else ("#f59e0b" if job.overall_status == StageStatus.WARNING else "#ef4444")
        st.markdown(
            f'<div style="background-color: {status_color}22; border: 1px solid {status_color}; border-radius: 8px; padding: 8px; text-align: center;">'
            f'<div style="font-size: 0.75rem; color: #94a3b8;">STATUS</div>'
            f'<div style="font-weight: 700; color: {status_color}; font-size: 0.95rem;">{job.overall_status.upper()}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.caption("ℹ️ **Page Accounting:** Physical Pages = Total pages in PDF file. Successfully Extracted Pages = Pages yielding clean text, tables, or diagrams. Indexed Chunks = Fine-grained retrieval units (dense pages yield multiple chunks, blank pages yield 0).")
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # --- Stage Breakdown Table ---
    st.markdown("### ⏱️ Stage-by-Stage Latency & Status")
    stage_data = []
    for sname, sinfo in job.stages.items():
        stage_data.append({
            "Stage": sname.replace("_", " ").title(),
            "Status": "✅ " + sinfo.status.upper() if sinfo.status == StageStatus.COMPLETED else ("⚠️ " + sinfo.status.upper() if sinfo.status == StageStatus.WARNING else "❌ " + sinfo.status.upper()),
            "Duration (ms)": f"{sinfo.duration_ms:.1f}",
            "Details Summary": str(sinfo.details)[:90] + ("..." if len(str(sinfo.details)) > 90 else "")
        })
    st.dataframe(stage_data, use_container_width=True, hide_index=True)

    # --- Suspicious Conditions & Heuristic Diagnostics ---
    st.markdown("### 🔎 Heuristic Anomaly Detection")
    st.caption("Automated quality heuristics identifying structural limitations. Flags distinguish confirmed errors from diagnostic warnings.")

    if not job.suspicious_conditions:
        st.success("✅ Clean Document: No unreadable pages, scanned image ratios, or extraction anomalies detected.")
    else:
        for idx, cond in enumerate(job.suspicious_conditions, 1):
            severity_icon = "⚠️" if cond.severity == "warning" else ("❌" if cond.severity == "error" else "ℹ️")
            border_color = "#f59e0b" if cond.severity == "warning" else ("#ef4444" if cond.severity == "error" else "#3b82f6")
            st.markdown(
                f'<div style="background-color: #18181b; border-left: 4px solid {border_color}; border-radius: 4px; padding: 10px 14px; margin-bottom: 8px;">'
                f'<div style="font-weight: 600; color: #f8fafc;">{severity_icon} Page {cond.page}: {cond.condition_type.replace("_", " ").title()} ({cond.severity.upper()})</div>'
                f'<div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 4px;">{cond.description}</div>'
                f'<div style="color: #64748b; font-size: 0.8rem; font-family: monospace; margin-top: 4px;">Heuristic Rule: {cond.heuristic_rule}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
