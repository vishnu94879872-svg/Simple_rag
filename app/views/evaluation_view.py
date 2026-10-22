"""Module G: RAG Evaluation Lab view with repeatable benchmarks and run comparisons."""

import streamlit as st
import io
from typing import List, Dict, Any

from observability.storage import storage
from observability.eval_lab import eval_lab
from observability.models import EvalRun, EvalBenchmarkItem
from app.ui import empty_state, page_header


def render_evaluation_lab(pipeline):
    """Render Module G - RAG Evaluation Lab."""
    page_header(
        "Module · Evaluation",
        "Evaluation Lab",
        "Run repeatable regression benchmarks, compute deterministic retrieval and generation metrics, and compare pipelines side by side.",
    )

    tab_run, tab_history, tab_datasets = st.tabs(["Run benchmark", "History & comparison", "Benchmark datasets"])

    benchmarks = storage.list_benchmarks()
    if not benchmarks:
        benchmarks = ["quantum_bench"]

    # --- TAB 1: Run Benchmark ---
    with tab_run:
        c1, c2, c3 = st.columns([1.5, 1, 1])
        with c1:
            selected_bench = st.selectbox("Benchmark Dataset", options=benchmarks, index=0)
        with c2:
            eval_strategy = st.selectbox("Pipeline Strategy", options=["reranked", "hybrid_rrf", "vector_only", "bm25_only"], index=0, key="eval_strat")
        with c3:
            eval_top_k = st.selectbox("Top-k Candidates", options=[3, 5, 10], index=1, key="eval_topk")

        c_limit, c_action = st.columns([1, 2.2], gap="medium", vertical_alignment="bottom")
        with c_limit:
            limit_items = st.number_input(
                "Limit evaluation items (0 for all)",
                min_value=0,
                max_value=100,
                value=5,
                help="Use a small limit for fast interactive benchmarking, or 0 for complete benchmark.",
                width="stretch",
            )
        with c_action:
            run_benchmark = st.button("Execute benchmark run", type="primary", width="stretch")

        if run_benchmark:
            with st.spinner(f"Running benchmark '{selected_bench}' under strategy '{eval_strategy}'..."):
                run = eval_lab.run_benchmark(
                    benchmark_name=selected_bench,
                    strategy=eval_strategy,
                    top_k=eval_top_k,
                    max_items=limit_items if limit_items > 0 else None
                )
            st.success(f"Benchmark run `{run.run_id}` completed successfully!")
            st.rerun()

    # --- TAB 2: Run History & Comparison ---
    with tab_history:
        runs = storage.list_eval_runs(limit=30)
        if not runs:
            empty_state(
                "No benchmark runs recorded",
                "Execute a benchmark run above to establish a baseline for recall, ranking quality, and groundedness.",
                mark="◫",
            )
        else:
            st.markdown("### Historical benchmark runs")
            summary_table = []
            for r in runs:
                summary_table.append({
                    "Run ID": r.run_id,
                    "Benchmark": f"{r.benchmark_name} ({getattr(r, 'dataset_version', 'v2.0')})",
                    "Strategy": r.strategy,
                    "Top-k": r.top_k,
                    "Recall@k": f"{r.recall_at_k:.1f}%",
                    "Precision@k": f"{r.precision_at_k:.1f}%",
                    "MRR": f"{r.mrr:.3f}",
                    "Correctness": f"{r.correctness_score:.1f}%",
                    "Citation Acc": f"{r.citation_accuracy:.1f}%",
                    "Groundedness": f"{r.groundedness_score:.1f}%",
                    "Abstention Acc": f"{r.abstention_accuracy:.1f}%",
                    "Latency (ms)": f"{r.avg_latency_ms:.0f}"
                })
            st.dataframe(summary_table, width="stretch", hide_index=True)

            # Side-by-side comparison
            st.divider()
            st.markdown("### Side-by-side run comparator")
            st.caption("Side-by-side statistical comparison. Improvements are verified against empirical measurements only.")

            if len(runs) >= 2:
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    run1_id = st.selectbox("Baseline Run (Reference):", options=[r.run_id for r in runs], index=1)
                with col_r2:
                    run2_id = st.selectbox("Candidate Run (Experimental):", options=[r.run_id for r in runs], index=0)

                run1 = storage.get_eval_run(run1_id)
                run2 = storage.get_eval_run(run2_id)

                if run1 and run2:
                    def calc_delta(val2, val1, is_latency=False):
                        diff = val2 - val1
                        sign = "+" if diff > 0 else ""
                        if abs(diff) < 0.001:
                            return "0.00 (Unchanged)"
                        if is_latency:
                            verdict = "Faster" if diff < 0 else "Slower"
                            return f"{sign}{diff:.1f} ms ({verdict})"
                        else:
                            verdict = "Improved" if diff > 0 else "Regressed"
                            return f"{sign}{diff:.2f}% ({verdict})"

                    diff_metrics = [
                        {"Metric": "Recall@k", f"Baseline ({run1_id})": f"{run1.recall_at_k:.1f}%", f"Candidate ({run2_id})": f"{run2.recall_at_k:.1f}%", "Delta": calc_delta(run2.recall_at_k, run1.recall_at_k)},
                        {"Metric": "Precision@k", f"Baseline ({run1_id})": f"{run1.precision_at_k:.1f}%", f"Candidate ({run2_id})": f"{run2.precision_at_k:.1f}%", "Delta": calc_delta(run2.precision_at_k, run1.precision_at_k)},
                        {"Metric": "MRR", f"Baseline ({run1_id})": f"{run1.mrr:.3f}", f"Candidate ({run2_id})": f"{run2.mrr:.3f}", "Delta": f"{run2.mrr - run1.mrr:+.4f}"},
                        {"Metric": "Answer Correctness", f"Baseline ({run1_id})": f"{run1.correctness_score:.1f}%", f"Candidate ({run2_id})": f"{run2.correctness_score:.1f}%", "Delta": calc_delta(run2.correctness_score, run1.correctness_score)},
                        {"Metric": "Citation Accuracy", f"Baseline ({run1_id})": f"{run1.citation_accuracy:.1f}%", f"Candidate ({run2_id})": f"{run2.citation_accuracy:.1f}%", "Delta": calc_delta(run2.citation_accuracy, run1.citation_accuracy)},
                        {"Metric": "Groundedness", f"Baseline ({run1_id})": f"{run1.groundedness_score:.1f}%", f"Candidate ({run2_id})": f"{run2.groundedness_score:.1f}%", "Delta": calc_delta(run2.groundedness_score, run1.groundedness_score)},
                        {"Metric": "Abstention Accuracy", f"Baseline ({run1_id})": f"{run1.abstention_accuracy:.1f}%", f"Candidate ({run2_id})": f"{run2.abstention_accuracy:.1f}%", "Delta": calc_delta(run2.abstention_accuracy, run1.abstention_accuracy)},
                        {"Metric": "Average Latency", f"Baseline ({run1_id})": f"{run1.avg_latency_ms:.0f} ms", f"Candidate ({run2_id})": f"{run2.avg_latency_ms:.0f} ms", "Delta": calc_delta(run2.avg_latency_ms, run1.avg_latency_ms, is_latency=True)},
                    ]
                    st.dataframe(diff_metrics, width="stretch", hide_index=True)

                    # Stage Latencies Breakdown comparison if present
                    if getattr(run1, 'stage_latencies_avg_ms', None) or getattr(run2, 'stage_latencies_avg_ms', None):
                        st.markdown("#### Stage latency breakdown (ms)")
                        stg_keys = set(list(getattr(run1, 'stage_latencies_avg_ms', {}).keys()) + list(getattr(run2, 'stage_latencies_avg_ms', {}).keys()))
                        stg_rows = []
                        for sk in sorted(stg_keys):
                            s1 = getattr(run1, 'stage_latencies_avg_ms', {}).get(sk, 0.0)
                            s2 = getattr(run2, 'stage_latencies_avg_ms', {}).get(sk, 0.0)
                            stg_rows.append({
                                "Pipeline Stage": sk.replace("_", " ").title(),
                                f"Baseline ({run1_id})": f"{s1:.2f} ms" if s1 else "N/A",
                                f"Candidate ({run2_id})": f"{s2:.2f} ms" if s2 else "N/A",
                                "Delta": f"{s2 - s1:+.2f} ms"
                            })
                        if stg_rows:
                            st.dataframe(stg_rows, width="stretch", hide_index=True)

    # --- TAB 3: Benchmark Datasets ---
    with tab_datasets:
        st.markdown("### Dataset management")
        curr_bench = st.selectbox("Select Dataset to Inspect / Export:", options=benchmarks, index=0, key="inspect_bench")
        items = storage.load_benchmark(curr_bench)

        # Download CSV
        csv_data = eval_lab.export_benchmark_to_csv(items)
        st.download_button(
            label="Export dataset as CSV",
            data=csv_data,
            file_name=f"{curr_bench}.csv",
            mime="text/csv"
        )

        # Upload CSV
        st.divider()
        st.markdown("#### Import custom benchmark (CSV)")
        uploaded_csv = st.file_uploader("Upload CSV Benchmark (columns: id, question, expected_pages, topic, should_abstain)", type=["csv"])
        new_bench_name = st.text_input("New Benchmark Name:", value="custom_bench")
        if uploaded_csv and st.button("Import and Save Benchmark"):
            csv_str = uploaded_csv.getvalue().decode("utf-8")
            new_items = eval_lab.import_benchmark_from_csv(csv_str)
            storage.save_benchmark(new_bench_name, new_items)
            st.success(f"Imported {len(new_items)} benchmark items into '{new_bench_name}'!")
            st.rerun()

        st.divider()
        st.markdown(f"#### Items in `{curr_bench}` ({len(items)} items)")
        items_table = [
            {
                "ID": it.id,
                "Question": it.question,
                "Expected Pages": str(it.expected_pages),
                "Topic": it.topic or "",
                "Should Abstain": "Yes" if it.should_abstain else "No"
            }
            for it in items
        ]
        st.dataframe(items_table, width="stretch", hide_index=True)
