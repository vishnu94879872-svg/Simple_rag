"""Module A: RAG Observability Dashboard view with authoritative backend telemetry."""

import streamlit as st

from config import settings
from observability.storage import storage
from observability.telemetry import authoritative_telemetry
from app.ui import empty_state, hero, pill


def _storage_pill() -> str:
    status = storage.get_status()
    if status["backend"] == "supabase" and not status.get("degraded"):
        return pill("Supabase connected", "success", dot=True)
    if status["backend"] == "supabase":
        return pill("Supabase degraded", "warning", dot=True)
    if status.get("last_error"):
        return pill("Local fallback", "warning", dot=True)
    return pill("Local persistence", "default", dot=True)


def render_overview(pipeline):
    """Render Module A - RAG Observability Dashboard with verified backend metrics."""
    metrics = authoritative_telemetry.get_dashboard_metrics()
    bm25_count = pipeline.retrieval_engine.bm25_store.count()

    hero(
        title="See exactly where your RAG pipeline breaks.",
        description=(
            "Trace, evaluate, and diagnose every stage from PDF parsing to grounded citation — "
            "against live index, latency, and failure telemetry rather than intuition."
        ),
        flow=["Ingest", "Parse", "Chunk", "Embed", "Hybrid retrieval", "Rerank", "Synthesize", "Audit"],
        status_pills=[
            _storage_pill(),
            pill(f"{settings.llm_provider} · {settings.llm_model}", "accent"),
            pill(f"{metrics['total_chunks']:,} vectors indexed"),
            pill(f"{bm25_count:,} BM25 docs"),
        ],
    )

    # --- Top Row: Storage & Corpus Metrics ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Indexed Documents",
            value=f"{metrics['indexed_documents_count']}",
            help=f"PDF book files verified in data/books/: {', '.join(f['filename'] for f in metrics['pdf_files'])}",
        )
    with c2:
        pages_val = metrics["distinct_indexed_pages"]
        st.metric(
            label="Extracted Pages",
            value=f"{pages_val}" if pages_val > 0 else "—",
            delta=f"{metrics['physical_pdf_pages']} physical pages across PDFs",
            delta_color="off",
            help="Distinct pages with indexed chunks in ChromaDB metadata, cross-referenced with physical PDF pages.",
        )
    with c3:
        st.metric(
            label="Indexed Chunks",
            value=f"{metrics['total_chunks']}",
            delta=f"BM25: {bm25_count}",
            delta_color="off",
            help="Total vectors persisted in ChromaDB and tokenized documents in the BM25 index.",
        )
    with c4:
        st.metric(
            label="Ingestion Jobs",
            value=f"{metrics['total_jobs']}",
            delta=metrics["jobs_delta"],
            delta_color=metrics["jobs_delta_color"],
            help="Telemetry-tracked ingestion runs. If 0, documents were pre-indexed prior to instrumentation.",
        )

    # --- Second Row: Query & Latency Telemetry ---
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric(
            label="Questions Processed",
            value=f"{metrics['questions_processed']}",
            help="Total user questions with execution traces in storage.",
        )
    with q2:
        st.metric(
            label="Retrieval Failure Rate",
            value=f"{metrics['retrieval_failure_rate']:.1f}%",
            delta=f"{metrics['failed_retrieval_count']} zero-hit queries" if metrics["failed_retrieval_count"] > 0 else "0 misses",
            delta_color="inverse" if metrics["retrieval_failure_rate"] > 0 else "normal",
            help="Percentage of traced queries where both dense and sparse retrieval returned 0 candidates.",
        )
    with q3:
        st.metric(
            label="Avg Response Latency",
            value=metrics["avg_latency_str"],
            delta=metrics["avg_latency_note"],
            delta_color="off",
            help="Arithmetic mean duration across all recorded pipeline traces.",
        )
    with q4:
        st.metric(
            label="P95 Latency",
            value=metrics["p95_latency_str"],
            delta=metrics["p95_latency_note"],
            delta_color="off",
            help="95th percentile latency. Requires at least 5 measurements for a meaningful percentile.",
        )

    # --- Corpus Breakdown ---
    with st.expander("Document corpus breakdown (database audit)", expanded=False):
        if not metrics["books_breakdown"]:
            st.caption("No chunk metadata currently recorded in ChromaDB.")
        else:
            book_table = []
            for title, info in metrics["books_breakdown"].items():
                p_min, p_max = info.get("page_range", (1, 1))
                book_table.append(
                    {
                        "Document Title": title,
                        "Distinct Indexed Pages": info.get("distinct_pages", 0),
                        "Page Span": f"Page {p_min} to {p_max}",
                        "Indexed Chunks": info.get("chunks", 0),
                    }
                )
            st.dataframe(book_table, width="stretch", hide_index=True)

    st.divider()

    # --- Filter Controls ---
    st.markdown("### Filter telemetry records")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        doc_filter = st.selectbox(
            "Filter by document",
            options=["All Documents"] + [f["filename"] for f in metrics["pdf_files"]],
            index=0,
        )
    with fc2:
        status_filter = st.selectbox("Filter by trace status", options=["All Statuses", "success", "error", "0 hits"], index=0)
    with fc3:
        eval_runs = storage.list_eval_runs(limit=50)
        eval_run_filter = st.selectbox(
            "Filter by evaluation run",
            options=["All Runs"] + [r.run_id for r in eval_runs],
            index=0,
        )
    traces = storage.list_traces(limit=200)
    filtered_traces = traces
    if status_filter == "success":
        filtered_traces = [t for t in filtered_traces if t.status == "success" and len(t.retrieved_chunk_ids) > 0]
    elif status_filter == "error":
        filtered_traces = [t for t in filtered_traces if t.status == "error"]
    elif status_filter == "0 hits":
        filtered_traces = [t for t in filtered_traces if len(t.retrieved_chunk_ids) == 0]

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Recent pipeline traces")
        if not filtered_traces:
            empty_state(
                "No traces match these filters",
                "Ask a question in Book Tutor, or clear the status filter, to populate the execution log.",
                mark="⌁",
            )
        else:
            trace_table_data = []
            for t in filtered_traces[:15]:
                trace_table_data.append(
                    {
                        "Trace ID": t.trace_id,
                        "Query": t.query[:45] + ("..." if len(t.query) > 45 else ""),
                        "Strategy": t.strategy,
                        "Chunks": len(t.retrieved_chunk_ids),
                        "Latency (ms)": f"{t.total_duration_ms:.0f}",
                        "Status": t.status.upper(),
                    }
                )
            st.dataframe(trace_table_data, width="stretch", hide_index=True)

    with col_right:
        st.markdown("### Benchmark evaluation history")
        if not eval_runs:
            empty_state(
                "No evaluation runs yet",
                "Open Evaluation Lab to score Recall@k, MRR, groundedness, and citation accuracy on the bundled benchmark.",
                mark="◫",
            )
        else:
            visible_runs = [r for r in eval_runs if eval_run_filter == "All Runs" or r.run_id == eval_run_filter]
            eval_table_data = []
            for r in visible_runs[:15]:
                eval_table_data.append(
                    {
                        "Run ID": r.run_id,
                        "Benchmark": r.benchmark_name,
                        "Recall@k": f"{r.recall_at_k:.1f}%",
                        "MRR": f"{r.mrr:.3f}",
                        "Citation Acc": f"{r.citation_accuracy:.1f}%",
                        "Avg Latency": f"{r.avg_latency_ms:.0f} ms",
                    }
                )
            st.dataframe(eval_table_data, width="stretch", hide_index=True)

    # --- Ingestion Jobs ---
    st.divider()
    st.markdown("### Ingestion job records")
    jobs = storage.list_jobs(limit=50)
    if doc_filter != "All Documents":
        jobs = [j for j in jobs if j.filename == doc_filter]

    if not jobs:
        empty_state(
            "No monitored ingestion jobs",
            "Existing books in data/books were indexed before telemetry was enabled. Upload a PDF in Ingestion Monitor to record live stage diagnostics.",
            mark="⇧",
        )
    else:
        job_summary = [
            {
                "Job ID": j.job_id,
                "Filename": j.filename,
                "Pages": j.page_count,
                "Chunks": j.chunk_count,
                "Status": j.overall_status.upper(),
                "Warnings": len(j.suspicious_conditions),
            }
            for j in jobs
        ]
        st.dataframe(job_summary, width="stretch", hide_index=True)
