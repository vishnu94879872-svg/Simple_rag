"""Module E: Prompt and LLM Trace Viewer view."""

import streamlit as st
from typing import List, Dict, Any, Optional

from observability.storage import storage
from observability.models import PipelineTrace


def render_trace_viewer():
    """Render Module E - Prompt and LLM Trace Viewer."""
    st.markdown("## ⏱️ Prompt & LLM Trace Viewer")
    st.caption("End-to-end execution waterfall tracing request ingress, query rewriting, candidate retrieval, prompt assembly, and model completion.")

    traces = storage.list_traces(limit=100)
    if not traces:
        st.info("No query traces recorded yet. Submit a question in the **Book Tutor** or execute a query in the **Retrieval Playground** to generate traces.")
        return

    # Select trace
    trace_options = {
        f"{t.trace_id} — '{t.query[:35]}...' ({t.total_duration_ms:.0f}ms, {t.status})": t
        for t in traces
    }
    selected_key = st.selectbox("Select Execution Trace to Inspect:", options=list(trace_options.keys()))
    trace: PipelineTrace = trace_options[selected_key]

    # --- Header Metrics ---
    st.markdown("### 🔍 Trace Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Correlation ID", trace.trace_id)
    with m2:
        st.metric("Total Latency", f"{trace.total_duration_ms:.1f} ms")
    with m3:
        st.metric("Strategy / Top-k", f"{trace.strategy} (k={trace.top_k})")
    with m4:
        st.metric("Model / Provider", f"{trace.model} ({trace.provider})")
    with m5:
        tokens_total = trace.token_usage.get("total_tokens", 0)
        cost_str = f"${trace.estimated_cost_usd:.5f}" if trace.estimated_cost_usd is not None else "N/A"
        st.metric("Tokens / Est. Cost", f"{tokens_total} tok", delta=cost_str)

    st.markdown(f"**Original Query:** `{trace.query}`")
    if trace.rewritten_query:
        st.markdown(f"**Rewritten Query:** `{trace.rewritten_query}`")

    st.divider()

    # --- Execution Waterfall / Step Breakdown ---
    st.markdown("### 🌊 Pipeline Stage Trace Waterfall")
    st.caption("Visual chronological waterfall tracing the real sequence of executed pipeline stages.")

    if not trace.steps:
        st.info("No atomic sub-steps recorded in this trace.")
    else:
        # Calculate timeline offsets for Gantt visualization
        total_time = max(trace.total_duration_ms, sum(s.duration_ms for s in trace.steps), 0.001)
        running_time = 0.0

        for idx, step in enumerate(trace.steps, 1):
            dur = step.duration_ms
            step_left_pct = min(100.0, max(0.0, (running_time / total_time) * 100.0))
            step_width_pct = min(100.0 - step_left_pct, max(1.5, (dur / total_time) * 100.0))
            running_time += dur

            status = step.status.lower()
            if status == "success":
                status_icon = "✅"
                badge_color = "#10b981"
                status_text = "Completed"
            elif status == "skipped":
                status_icon = "⏭️"
                badge_color = "#94a3b8"
                status_text = "Skipped"
            elif status == "warning":
                status_icon = "⚠️"
                badge_color = "#f59e0b"
                status_text = "Warning"
            else:
                status_icon = "❌"
                badge_color = "#ef4444"
                status_text = "Failed"

            stage_display_name = step.name.replace("_", " ").title()

            # Render stage container with interactive waterfall bar
            with st.container():
                st.markdown(
                    f'<div style="background-color: #18181b; border: 1px solid #27272a; border-left: 4px solid {badge_color}; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
                    f'<div>'
                    f'<span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">{status_icon} Stage {idx}: {stage_display_name}</span> '
                    f'<span style="background-color: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}55; padding: 1px 7px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; margin-left: 6px;">{status_text}</span>'
                    f'</div>'
                    f'<div style="display: flex; align-items: center; gap: 10px;">'
                    f'<span style="color: #94a3b8; font-size: 0.8rem; font-family: monospace;">{step.timestamp}</span>'
                    f'<span style="background-color: #27272a; color: #60a5fa; font-family: monospace; font-weight: 600; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem;">{step.duration_ms:.2f} ms</span>'
                    f'</div>'
                    f'</div>'
                    # Visual Waterfall Progress Track
                    f'<div style="width: 100%; height: 6px; background-color: #27272a; border-radius: 3px; position: relative; overflow: hidden; margin: 4px 0 2px 0;">'
                    f'<div style="position: absolute; left: {step_left_pct:.1f}%; width: {step_width_pct:.1f}%; height: 100%; background-color: {badge_color}; border-radius: 3px;"></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                with st.expander(f"Inspect Stage {idx} Payload: {stage_display_name}", expanded=(step.status in ["failed", "warning"])):
                    c_in, c_out = st.columns(2)
                    with c_in:
                        st.markdown("**📥 Input Summary:**")
                        st.json(step.input_summary if step.input_summary else {"input": "None"})
                    with c_out:
                        st.markdown("**📤 Output Summary:**")
                        st.json(step.output_summary if step.output_summary else {"output": "None"})
                    if step.error:
                        st.error(f"❌ Execution Error: {step.error}")

    st.divider()

    # --- Developer View: Prompt & Context Inspect ---
    st.markdown("### 🛡️ Developer Prompt & Retrieved Evidence View")
    st.caption("Inspect exact context and user prompt passed to the LLM. All secrets and sensitive tokens are strictly redacted.")

    tab_prompt, tab_ctx, tab_ans = st.tabs(["💬 Full Prompt", "📚 Retrieved Context", "🎯 Model Answer"])

    with tab_prompt:
        if trace.prompt_preview:
            st.code(trace.prompt_preview, language="markdown")
        else:
            st.info("Prompt preview was not logged or was redacted by active privacy policy.")

    with tab_ctx:
        if trace.context_preview:
            st.code(trace.context_preview, language="markdown")
        else:
            st.info("Context preview not captured.")

    with tab_ans:
        st.markdown(trace.final_answer or "No answer generated.")
        if trace.citations:
            st.markdown("**Citations Attached:**")
            for cit in trace.citations:
                st.caption(f"- Chapter: {cit.get('chapter')}, Page: {cit.get('page')}")
