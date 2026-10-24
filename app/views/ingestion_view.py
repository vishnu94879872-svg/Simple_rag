"""Module B: Document Ingestion Diagnostics view."""

import streamlit as st
from pathlib import Path

from config import settings
from observability.storage import storage
from observability.ingestion_monitor import ingestion_monitor
from observability.models import IngestionJob, StageStatus, IngestionStageInfo
from app.ui import empty_state, escape, page_header, pill


def _format_stage_details(details) -> str:
    if not details:
        return "—"
    parts = []
    for key, value in details.items():
        label = key.replace("_", " ").capitalize()
        text = str(value)
        parts.append(f"{label}: {text}")
    joined = " · ".join(parts)
    return joined[:140] + ("…" if len(joined) > 140 else "")


def render_ingestion_monitor():
    """Render Module B - Document Ingestion Diagnostics."""
    page_header(
        "Module · Ingestion",
        "Ingestion Monitor",
        "Inspect PDF parsing, structural segmentation, chunk generation, and index synchronization stage by stage.",
    )

    # Top Ingestion Action / Upload
    with st.expander("Ingest and monitor a new PDF", expanded=False):
        uploaded_file = st.file_uploader("Upload PDF Book to Ingest & Benchmark", type=["pdf"])
        if uploaded_file is not None:
            if st.button("Run diagnostic ingestion", type="primary", width="stretch"):
                save_path = settings.data_dir / "books" / uploaded_file.name
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.status(f"Ingesting '{uploaded_file.name}' through monitored pipeline...", expanded=True) as status:
                    st.write("Running structural pre-flight inspection...")
                    job = ingestion_monitor.run_monitored_ingestion(save_path)
                    st.cache_resource.clear()
                    if job.overall_status == StageStatus.COMPLETED:
                        status.update(label=f"Successfully indexed '{uploaded_file.name}'", state="complete")
                    elif job.overall_status == StageStatus.WARNING:
                        status.update(label=f"Ingested '{uploaded_file.name}' with heuristic warnings", state="complete")
                    else:
                        status.update(label=f"Ingestion failed for '{uploaded_file.name}'", state="error")
                st.rerun()

    # Ingest / Sync existing book from data/books/
    book_files = list(Path(settings.data_dir / "books").glob("*.pdf")) if (settings.data_dir / "books").exists() else []
    if book_files:
        with st.expander("Ingest or re-index an existing PDF in data/books/", expanded=False):
            col_pdf, col_action = st.columns([2.4, 1], gap="medium", vertical_alignment="bottom")
            with col_pdf:
                selected_pdf = st.selectbox("Select document to ingest or re-index:", options=[b.name for b in book_files], key="existing_book_select", width="stretch")
            with col_action:
                run_diagnostics = st.button("Ingest selected PDF", key="btn_ingest_existing", width="stretch")
            if run_diagnostics:
                target = settings.data_dir / "books" / selected_pdf
                with st.status(f"Ingesting '{selected_pdf}' into storage...", expanded=True) as status:
                    job = ingestion_monitor.run_monitored_ingestion(target)
                    st.cache_resource.clear()
                    if job.overall_status == StageStatus.COMPLETED:
                        status.update(label=f"Successfully indexed '{selected_pdf}'", state="complete")
                    elif job.overall_status == StageStatus.WARNING:
                        status.update(label=f"Ingested '{selected_pdf}' with warnings", state="complete")
                    else:
                        status.update(label=f"Ingestion failed for '{selected_pdf}'", state="error")
                st.rerun()

    # List recorded jobs
    jobs = storage.list_jobs(limit=50)

    if not jobs:
        empty_state(
            "No ingestion jobs recorded yet",
            "Upload a PDF or run diagnostics on an existing book to capture stage-level telemetry.",
            mark="⇧",
        )
        return

    # Job selection
    job_options = {f"{j.filename} ({j.job_id}) - {j.overall_status.upper()}": j for j in jobs}
    selected_label = st.selectbox("Select ingestion job to inspect", options=list(job_options.keys()))
    job: IngestionJob = job_options[selected_label]

    # --- Job Header Summary ---
    st.markdown("### Document metadata & execution summary")
    
    struct_details = job.stages.get("structural_inspection", IngestionStageInfo(stage_name="")).details or {}
    phys_pages = struct_details.get("physical_pages", job.page_count)
    extracted_pages = struct_details.get("successfully_extracted_pages", job.page_count)
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(
            f'<div class="br-stage" style="min-height:102px;display:flex;flex-direction:column;justify-content:center;gap:6px;">'
            f'<div class="br-status-key">Document ID</div>'
            f'<div class="br-doc-id">{escape(job.document_id)}</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.metric("File Size", f"{job.file_size_bytes / 1024:.1f} KB")
    with m3:
        st.metric("Physical Pages", phys_pages, help="Total physical page count in source PDF file")
    with m4:
        st.metric("Extracted Pages", extracted_pages, help="Pages successfully yielding readable text, tables, or drawings")
    with m5:
        st.metric("Indexed Chunks", job.chunk_count, help="Searchable retrieval chunks after recursive sentence-safe splitting")
    with m6:
        tone = "success" if job.overall_status == StageStatus.COMPLETED else ("warning" if job.overall_status == StageStatus.WARNING else "danger")
        st.markdown(
            f'<div class="br-stage" style="min-height:102px;display:flex;flex-direction:column;justify-content:center;align-items:center;gap:8px;">'
            f'<div class="br-status-key">Status</div>{pill(job.overall_status.value, tone, dot=True)}</div>',
            unsafe_allow_html=True,
        )

    st.caption("Page accounting: physical pages = total pages in the PDF; extracted pages = pages yielding clean text, tables, or diagrams; indexed chunks = fine-grained retrieval units.")
    st.markdown("### Stage-by-stage latency & status")
    stage_data = []
    for sname, sinfo in job.stages.items():
        stage_data.append({
            "Stage": sname.replace("_", " ").title(),
            "Status": sinfo.status.upper(),
            "Duration (ms)": f"{sinfo.duration_ms:.1f}",
            "Details Summary": _format_stage_details(sinfo.details),
        })
    st.dataframe(stage_data, width="stretch", hide_index=True)

    # --- Suspicious Conditions & Heuristic Diagnostics ---
    st.markdown("### Heuristic anomaly detection")
    st.caption("Automated quality heuristics identifying structural limitations. Confirmed errors are separated from diagnostic warnings.")

    if not job.suspicious_conditions:
        st.success("Clean document: no unreadable pages, scanned image ratios, or extraction anomalies detected.")
    else:
        for cond in job.suspicious_conditions:
            tone = "warning" if cond.severity == "warning" else ("danger" if cond.severity == "error" else "info")
            st.markdown(
                f'<div class="br-verdict br-verdict--{tone}">'
                f'<div class="br-verdict-title">Page {cond.page} · {escape(cond.condition_type.replace("_", " ").title())} · {escape(cond.severity.upper())}</div>'
                f'<div class="br-verdict-body">{escape(cond.description)}</div>'
                f'<div style="color:var(--br-text-dim);font-family:var(--br-mono);font-size:10px;margin-top:6px;">RULE · {escape(cond.heuristic_rule)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
