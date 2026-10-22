"""Module I: Controlled Experiments Workbench view."""

import time
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any

from config import settings
from observability.storage import storage
from observability.experiment_workbench import chunk_experiment_workbench
from observability.models import ChunkExperimentResult
from app.ui import empty_state, page_header


def render_experiment_workbench():
    """Render Module I - Controlled Experiments Workbench."""
    page_header(
        "Module · Experiments",
        "Chunking Experiments",
        "Benchmark alternative chunking and retrieval configurations in isolated sandboxes that never touch the production index.",
    )

    tab_chunk, tab_retrieval = st.tabs(["Chunking sandbox", "Retrieval matrix"])

    # --- TAB 1: Chunking Experiments ---
    with tab_chunk:
        book_files = list(Path(settings.data_dir / "books").glob("*.pdf")) if (settings.data_dir / "books").exists() else []
        if not book_files:
            empty_state(
                "No PDF books available",
                "Add a PDF to data/books/ or ingest one from the Ingestion Monitor to run chunking experiments.",
                mark="▤",
            )
        else:
            # Configuration controls
            c_pdf, c_strat = st.columns([1.5, 1.5])
            with c_pdf:
                selected_pdf = st.selectbox("Select Target PDF Book:", options=[b.name for b in book_files], index=0)
                pdf_path = settings.data_dir / "books" / selected_pdf
            with c_strat:
                strategy_name = st.selectbox(
                    "Chunking Strategy",
                    options=["heading_aware", "sentence_aware", "paragraph_aware", "fixed_size", "fixed_overlap"],
                    index=0,
                    help="Compare existing standard chunking vs hierarchical heading-aware chunking"
                )

            c_size, c_overlap, c_ret = st.columns(3)
            with c_size:
                chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=800, step=50)
            with c_overlap:
                chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=150, step=25)
            with c_ret:
                ret_strat = st.selectbox("Benchmark Retrieval Strategy:", ["reranked", "hybrid_rrf", "vector_only", "bm25_only"], index=0)

            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                run_text_exp = st.button("Run text splitter experiment", width="stretch")
            with c_btn2:
                run_bench_exp = st.button("Run reproducible benchmark", type="primary", width="stretch", help="Indexes chunks in an isolated sandbox and benchmarks Recall@k, groundedness, citation accuracy, and latency against quantum_bench.")

            if run_text_exp:
                with st.spinner(f"Evaluating '{strategy_name}' chunking on '{selected_pdf}'..."):
                    exp_result = chunk_experiment_workbench.run_experiment(
                        pdf_path=pdf_path,
                        strategy_name=strategy_name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                st.success(f"Experiment `{exp_result.experiment_id}` completed successfully in {exp_result.processing_duration_ms:.1f}ms! (Production index unchanged)")
                st.rerun()

            if run_bench_exp:
                with st.spinner(f"Building isolated index and running benchmark experiment..."):
                    bench_result = chunk_experiment_workbench.run_benchmark_experiment(
                        pdf_path=pdf_path,
                        benchmark_name="quantum_bench",
                        strategy_name=strategy_name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        retrieval_strategy=ret_strat,
                        top_k=5,
                        max_queries=8
                    )
                st.success(f"Benchmark experiment `{bench_result.experiment_id}` completed! Recall@5: {bench_result.recall_at_k * 100:.1f}%, MRR: {bench_result.mrr:.3f}, Latency: {bench_result.avg_latency_ms:.1f}ms (Production index untouched)")
                st.rerun()

            # List previous experiments
            experiments = storage.list_experiments(limit=25)
            if experiments:
                st.divider()
                st.markdown("### Reproducible experiment history")
                exp_table = []
                for exp in experiments:
                    dist = exp.size_distribution
                    exp_table.append({
                        "Experiment ID": exp.experiment_id,
                        "Strategy": exp.strategy_name,
                        "Size": exp.chunk_size,
                        "Overlap": exp.chunk_overlap,
                        "Chunks": exp.total_chunks,
                        "Retrieval Strat": exp.retrieval_strategy or "N/A (Text Only)",
                        "Recall@5": f"{exp.recall_at_k * 100:.1f}%" if exp.recall_at_k is not None else "—",
                        "MRR": f"{exp.mrr:.3f}" if exp.mrr is not None else "—",
                        "Groundedness": f"{exp.groundedness_score * 100:.1f}%" if exp.groundedness_score is not None else "—",
                        "Citation Acc": f"{exp.citation_accuracy * 100:.1f}%" if exp.citation_accuracy is not None else "—",
                        "Avg Latency": f"{exp.avg_latency_ms:.1f} ms" if exp.avg_latency_ms is not None else f"{exp.processing_duration_ms:.0f} ms",
                        "Sandbox Index": "Isolated" if exp.is_experimental_index else "Active"
                    })
                st.dataframe(exp_table, width="stretch", hide_index=True)

                # Sample chunk inspector from most recent experiment
                latest = experiments[0]
                st.markdown(f"#### Sample chunks from latest run · `{latest.experiment_id}` ({latest.strategy_name})")
                for sample in latest.sample_chunks:
                    st.markdown(f"**Sample Chunk #{sample['index']}** (*{sample['chars']} chars*):")
                    st.code(sample["preview"], language="markdown")

    # --- TAB 2: Retrieval Strategy & Top-K Matrix ---
    with tab_retrieval:
        st.markdown("### Retrieval strategy & top-k matrix")
        st.caption("Benchmark Vector vs Keyword (BM25) vs Hybrid RRF vs Reranked across top-k settings (3, 5, 10).")

        c_strat_sel, c_k_sel = st.columns(2)
        with c_strat_sel:
            eval_strategies = st.multiselect(
                "Retrieval Strategies to Compare:",
                options=["vector_only", "bm25_only", "hybrid_rrf", "reranked"],
                default=["vector_only", "bm25_only", "hybrid_rrf", "reranked"]
            )
        with c_k_sel:
            eval_ks = st.multiselect(
                "Top-k Candidate Counts to Test:",
                options=[3, 5, 10],
                default=[3, 5]
            )

        test_questions = [
            {"q": "What is a qubit in quantum computing?", "exp_pages": [1]},
            {"q": "How does Shor's algorithm challenge RSA?", "exp_pages": [2]},
            {"q": "What is quantum superposition?", "exp_pages": [1]},
        ]

        st.caption(f"Evaluation test suite: {len(test_questions)} canonical verification queries across {len(eval_strategies)} strategies x {len(eval_ks)} top-k settings.")

        if st.button("Run retrieval comparison matrix", type="primary", width="stretch"):
            from retrieval.engine import RetrievalEngine
            engine = RetrievalEngine()

            matrix_results = []
            progress_bar = st.progress(0.0)
            total_combinations = len(eval_strategies) * len(eval_ks)
            comb_idx = 0

            for strat in eval_strategies:
                for k_val in eval_ks:
                    hits_found = 0
                    reciprocal_ranks = []
                    latencies = []

                    for tq in test_questions:
                        t0 = time.time()
                        res = engine.retrieve(query=tq["q"], top_k=k_val, strategy=strat)
                        latencies.append((time.time() - t0) * 1000.0)

                        ret_pages = [int(h.get("metadata", {}).get("page", 0)) for h in res.hits]
                        first_rank = 0
                        for r_idx, p in enumerate(ret_pages, 1):
                            if p in tq["exp_pages"]:
                                first_rank = r_idx
                                break
                        if first_rank > 0:
                            hits_found += 1
                            reciprocal_ranks.append(1.0 / first_rank)
                        else:
                            reciprocal_ranks.append(0.0)

                    recall_at_k = (hits_found / len(test_questions)) * 100.0
                    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
                    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

                    matrix_results.append({
                        "Strategy": strat,
                        "Top-k": k_val,
                        "Reranking Used": "Yes" if strat == "reranked" else "No",
                        "Recall@k": f"{recall_at_k:.1f}%",
                        "MRR": f"{mrr:.3f}",
                        "Avg Retrieval Latency": f"{avg_lat:.1f} ms"
                    })

                    comb_idx += 1
                    progress_bar.progress(comb_idx / total_combinations)

            st.success("Matrix evaluation completed successfully! Production configuration was preserved.")
            st.dataframe(matrix_results, width="stretch", hide_index=True)

