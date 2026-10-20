"""Module H: Failure Classification and Root-Cause Assistant view with complete failure investigation workflow."""

import streamlit as st
from typing import List, Dict, Any, Optional

from observability.storage import storage
from observability.failure_classifier import failure_classifier
from observability.citation_verifier import citation_verifier
from observability.models import (
    FailureDiagnosis,
    FailureInvestigationReport,
    PipelineTrace,
    EvalBenchmarkItem,
)


def render_root_cause_assistant():
    """Render Module H - Failure Classification and Root-Cause Assistant."""
    st.markdown("## 🩺 Complete Failure Investigation Workflow")
    st.caption("Investigate failed questions end-to-end: original query, ground truth, retrieved chunks, context, answer, citations, failure stage, evidence, fix, and rerun test.")

    traces = storage.list_traces(limit=100)
    benchmarks = storage.list_benchmarks()
    saved_investigations = storage.list_investigations(limit=25)

    tab_investigate, tab_saved, tab_manual = st.tabs([
        "🔬 New Failure Investigation",
        f"📜 Saved Investigations ({len(saved_investigations)})",
        "🧪 Heuristic Simulator"
    ])

    # =========================================================================
    # TAB 1: Complete Failure Investigation Workflow
    # =========================================================================
    with tab_investigate:
        st.markdown("#### 1. Select Failed Query or Benchmark Target")
        source_mode = st.radio("Investigation Source:", ["Recent Pipeline Traces", "Benchmark Questions", "Custom Failed Query"], horizontal=True)

        selected_query = ""
        expected_ans = ""
        expected_evidence = {}
        actual_chunks = []
        final_context = ""
        gen_answer = ""
        trace_obj = None

        if source_mode == "Recent Pipeline Traces":
            if not traces:
                st.info("No recorded traces found. Execute a query in Chat or CLI to record traces.")
            else:
                trace_opts = {
                    f"{t.trace_id} — '{t.query[:45]}...' ({t.status}, {len(t.retrieved_chunk_ids)} chunks)": t
                    for t in traces
                }
                sel_trace_key = st.selectbox("Select Trace to Investigate:", options=list(trace_opts.keys()))
                trace_obj = trace_opts[sel_trace_key]
                selected_query = trace_obj.query
                gen_answer = trace_obj.final_answer
                final_context = trace_obj.context_preview or ""
                
                # Reconstruct retrieved chunk summaries from trace
                actual_chunks = [
                    {
                        "chunk_id": cid,
                        "content": trace_obj.context_preview or "Retrieved chunk content preserved in trace context.",
                        "metadata": {"page": 1, "chapter": "Document Context"}
                    }
                    for cid in trace_obj.retrieved_chunk_ids
                ]

        elif source_mode == "Benchmark Questions":
            bench_name = st.selectbox("Select Benchmark Dataset:", options=benchmarks if benchmarks else ["quantum_bench"])
            bench_items = storage.load_benchmark(bench_name)
            if not bench_items:
                st.info("No benchmark questions in this dataset.")
            else:
                b_opts = {f"{it.id} — '{it.question}' (Topic: {it.topic or 'General'})": it for it in bench_items}
                sel_b_key = st.selectbox("Select Question to Investigate:", options=list(b_opts.keys()))
                b_item: EvalBenchmarkItem = b_opts[sel_b_key]
                selected_query = b_item.question
                expected_ans = b_item.expected_answer or "Ground-truth answer defined in benchmark."
                expected_evidence = {
                    "expected_pages": b_item.expected_pages,
                    "expected_document": b_item.expected_document,
                    "should_abstain": b_item.should_abstain,
                    "topic": b_item.topic
                }

        else:  # Custom Failed Query
            selected_query = st.text_input("Enter Query to Investigate:", value="What is quantum teleportation?")
            expected_ans = st.text_input("Expected Answer (optional ground truth):", value="Quantum teleportation transfers state without moving particles.")
            exp_p = st.number_input("Expected Page (optional):", min_value=0, max_value=500, value=1)
            if exp_p > 0:
                expected_evidence = {"expected_pages": [exp_p]}

        col_run_inv, _ = st.columns([1, 2])
        with col_run_inv:
            start_inv = st.button("🚀 Run Failure Investigation", key="btn_start_inv", type="primary", use_container_width=True)

        if start_inv and selected_query:
            with st.spinner("Analyzing retrieval, prompt assembly, and synthesis evidence..."):
                # If we don't have retrieved chunks yet (e.g. from benchmark selection), run retrieval to inspect live evidence
                if not actual_chunks:
                    from retrieval.engine import RetrievalEngine
                    eng = RetrievalEngine()
                    ret_res = eng.retrieve(query=selected_query, strategy="reranked", top_k=5)
                    actual_chunks = ret_res.hits
                    if not gen_answer:
                        from generation.qa_chain import BookRAGPipeline
                        pipeline = BookRAGPipeline(retrieval_engine=eng)
                        ans_res = pipeline.query(selected_query, top_k=5)
                        gen_answer = ans_res.answer
                        final_context = "\n---\n".join([c.get("content", "") for c in ans_res.retrieved_chunks])

                audit = citation_verifier.audit_response(
                    question=selected_query,
                    answer=gen_answer,
                    retrieved_chunks=actual_chunks
                )

                report: FailureInvestigationReport = failure_classifier.create_investigation_report(
                    query=selected_query,
                    answer=gen_answer,
                    retrieved_chunks=actual_chunks,
                    expected_answer=expected_ans or None,
                    expected_source_evidence=expected_evidence,
                    final_llm_context=final_context,
                    citations=audit.detected_citations,
                    trace=trace_obj,
                    audit=audit
                )

                st.session_state["active_investigation_report"] = report

        # Render active report if present
        if "active_investigation_report" in st.session_state:
            _render_complete_investigation_card(st.session_state["active_investigation_report"])

    # =========================================================================
    # TAB 2: Saved Investigations
    # =========================================================================
    with tab_saved:
        if not saved_investigations:
            st.info("No saved failure investigation reports yet. Run an investigation from Tab 1 to record one.")
        else:
            inv_opts = {f"{inv.investigation_id} — '{inv.query[:40]}' ({inv.likely_failure_stage.value.upper()})": inv for inv in saved_investigations}
            sel_inv_key = st.selectbox("Select Past Investigation Report:", options=list(inv_opts.keys()))
            selected_inv: FailureInvestigationReport = inv_opts[sel_inv_key]
            _render_complete_investigation_card(selected_inv)

    # =========================================================================
    # TAB 3: Heuristic Simulator
    # =========================================================================
    with tab_manual:
        st.caption("Simulate hypothetical failures to verify diagnostic rules across the 10 failure stages.")
        sim_query = st.text_input("Simulated Query:", value="What is Shor's algorithm complexity?")
        sim_answer = st.text_area("Simulated Answer:", value="Shor's algorithm runs in polynomial time. (Chapter 99, Page 999)")
        sim_has_chunks = st.checkbox("Simulate chunks were retrieved", value=True)
        sim_exp_page = st.number_input("Expected Page:", value=2)

        if st.button("Run Simulation Diagnosis"):
            mock_chunks = [{"chunk_id": "c1", "content": "Text", "metadata": {"page": 1, "chapter": "Chapter 1"}}] if sim_has_chunks else []
            audit = citation_verifier.audit_response(sim_query, sim_answer, mock_chunks)
            diag = failure_classifier.diagnose(
                query=sim_query,
                answer=sim_answer,
                retrieved_chunks=mock_chunks,
                audit=audit,
                expected_page=sim_exp_page
            )
            cat_name = diag.failure_stage.value.replace("_", " ").title()
            st.success(f"Diagnosed Category: {cat_name} (Confidence: {int(diag.confidence * 100)}%)")
            st.markdown(f"**Likely Cause:** {diag.likely_cause}")
            st.markdown(f"**Recommended Fix:** {diag.recommended_fix}")


def _render_complete_investigation_card(report: FailureInvestigationReport):
    """Render the complete 10-item failure investigation report with rerun capabilities."""
    stage_name = report.likely_failure_stage.value.replace("_", " ").title()
    conf_pct = int(report.confidence * 100)

    st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="background-color: #18181b; border: 1px solid #3b82f6; border-radius: 8px; padding: 16px 20px; margin-bottom: 15px;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
        f'<span style="font-weight: 700; color: #60a5fa; font-size: 1.2rem;">🎯 Diagnosed Failure Stage: {stage_name} Failure</span>'
        f'<div style="display: flex; gap: 8px;">'
        f'<span style="background-color: #1e3a8a; color: #93c5fd; padding: 3px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem;">Confidence: {conf_pct}%</span>'
        f'<span style="background-color: #27272a; color: #f59e0b; padding: 3px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem;">Uncertainty: {report.uncertainty}</span>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # 10 Required Items
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 1. Original Query")
        st.code(report.query, language="text")

        st.markdown("#### 2. Expected Answer & Ground-Truth Evidence")
        if report.expected_answer or report.expected_source_evidence:
            st.info(f"**Expected Answer:** {report.expected_answer or 'None specified'}\n\n"
                    f"**Ground Truth Evidence:** `{report.expected_source_evidence}`")
        else:
            st.caption("No ground-truth benchmark evidence specified for this query.")

        st.markdown(f"#### 3. Actual Retrieved Chunks ({len(report.actual_retrieved_chunks)})")
        if not report.actual_retrieved_chunks:
            st.error("❌ 0 candidates retrieved. (Retrieval failure)")
        else:
            for idx, c in enumerate(report.actual_retrieved_chunks[:4], 1):
                cid = c.get("chunk_id", f"c_{idx}")
                meta = c.get("metadata", {})
                score = c.get("rerank_score", c.get("score", 0.0))
                with st.expander(f"Chunk #{idx}: {cid} (Page {meta.get('page', '?')}, Score: {score:.3f})"):
                    st.text(c.get("content", "")[:300] + "...")

        st.markdown("#### 4. Final LLM Context")
        with st.expander("📄 View Context Sent to Model"):
            st.text_area("Prompt Context", value=report.final_llm_context or "No context provided", height=150)

        st.markdown("#### 5. Generated Answer & Citations")
        st.markdown(f"**Answer:** {report.generated_answer or '*(empty answer)*'}")
        if report.citations:
            st.caption(f"Citations: `{report.citations}`")

    with col_b:
        st.markdown("#### 6. Evaluation Results")
        st.json(report.evaluation_results)

        st.markdown("#### 7. Likely Failure Stage")
        st.warning(f"**Stage:** `{report.likely_failure_stage.value}`")

        st.markdown("#### 8. Supporting Evidence")
        for ev in report.supporting_evidence:
            st.markdown(f"- {ev}")

        st.markdown("#### 9. Recommended Fix")
        st.success(report.recommended_fix)

        st.markdown("#### 10. Reproducible Regression Test")
        st.code(report.regression_test_code, language="python")

    # Live Rerun Workflow
    st.divider()
    st.markdown("### ⚡ Live Rerun Test After Applying Fix")
    st.caption("Test alternative settings (e.g. increase top_k or switch retrieval strategy) and evaluate before-and-after live.")

    c_s, c_k, c_btn = st.columns([1.5, 1, 1.2])
    with c_s:
        rerun_strat = st.selectbox("Rerun Strategy:", ["reranked", "hybrid_rrf", "vector_only", "bm25_only"], index=0)
    with c_k:
        rerun_k = st.number_input("Rerun Top-K:", min_value=1, max_value=20, value=10)
    with c_btn:
        st.write("")
        st.write("")
        trigger_rerun = st.button("🚀 Rerun Test Now", type="primary", use_container_width=True)

    if trigger_rerun:
        with st.spinner("Executing live rerun benchmark..."):
            try:
                res = failure_classifier.rerun_investigation_test(
                    report_id=report.investigation_id,
                    modified_strategy=rerun_strat,
                    modified_top_k=int(rerun_k)
                )
                st.session_state[f"rerun_{report.investigation_id}"] = res
                st.success(f"Rerun complete! Outcome: {res['verdict']}")
            except Exception as e:
                st.error(f"Rerun failed: {e}")

    # Display Rerun comparison if available
    rerun_data = st.session_state.get(f"rerun_{report.investigation_id}") or report.rerun_result
    if rerun_data:
        st.markdown("#### ⚖️ Before & After Verification Comparison")
        comp_table = [
            {"Metric / State": "Retrieval Candidate Count", "Before Fix": f"{len(report.actual_retrieved_chunks)}", "After Rerun": f"{rerun_data['new_retrieved_count']}", "Status": "Improved" if rerun_data['new_retrieved_count'] >= len(report.actual_retrieved_chunks) else "Same"},
            {"Metric / State": "Invalid Citation Count", "Before Fix": f"{report.evaluation_results.get('invalid_citations_count', 0)}", "After Rerun": f"{rerun_data['invalid_citations_count']}", "Status": "Pass" if rerun_data['invalid_citations_count'] == 0 else "Fail"},
            {"Metric / State": "Target Page Recall", "Before Fix": "Missed" if report.likely_failure_stage.value in ("retrieval", "reranking") else "Found", "After Rerun": "Hit ✅" if rerun_data['recall_hit'] else "Miss ❌", "Status": "Pass" if rerun_data['recall_hit'] else "Investigate"},
            {"Metric / State": "Audit Verdict", "Before Fix": report.evaluation_results.get('audit_verdict', 'unverified'), "After Rerun": rerun_data['audit_verdict'], "Status": rerun_data['verdict']}
        ]
        st.dataframe(comp_table, use_container_width=True, hide_index=True)
        st.caption(f"**Updated Answer Generated:** {rerun_data['new_answer']}")
