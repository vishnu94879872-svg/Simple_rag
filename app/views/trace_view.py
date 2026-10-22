"""Module E: Prompt and LLM Trace Viewer view."""

import streamlit as st
from typing import List, Dict, Any, Optional

from observability.storage import storage
from observability.models import PipelineTrace
from app.ui import empty_state, escape, page_header, pill


def render_trace_viewer():
    """Render Module E - Prompt and LLM Trace Viewer."""
    page_header(
        "Module · Tracing",
        "Trace Viewer",
        "Follow a request end to end through ingress, query rewriting, retrieval, prompt assembly, and model completion.",
    )

    traces = storage.list_traces(limit=100)
    if not traces:
        empty_state(
            "No query traces recorded yet",
            "Submit a question in Book Tutor or run a query in the Retrieval Playground to generate execution traces.",
            mark="⌁",
        )
        return

    # Select trace
    trace_options = {
        f"{t.trace_id} — '{t.query[:35]}...' ({t.total_duration_ms:.0f}ms, {t.status})": t
        for t in traces
    }
    selected_key = st.selectbox("Select Execution Trace to Inspect:", options=list(trace_options.keys()))
    trace: PipelineTrace = trace_options[selected_key]

    # --- Header Metrics ---
    st.markdown("### Trace overview")
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:2px 0 14px;">'
        f'{pill("Correlation ID", "accent")}'
        f'<code style="font-size:11px;color:var(--br-text-muted);">{escape(trace.trace_id)}</code>'
        f'{pill(trace.status, "success" if trace.status == "success" else "danger", dot=True)}'
        f"</div>",
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total latency", f"{trace.total_duration_ms:.1f} ms")
    with m2:
        st.metric("Strategy", trace.strategy, delta=f"top-k {trace.top_k}", delta_color="off")
    with m3:
        st.metric("Model", trace.model)
    with m4:
        st.metric("Provider", trace.provider)
    with m5:
        tokens_total = trace.token_usage.get("total_tokens", 0)
        cost_str = f"${trace.estimated_cost_usd:.5f}" if trace.estimated_cost_usd is not None else "n/a"
        st.metric("Tokens / est. cost", f"{tokens_total:,}", delta=cost_str)

    st.markdown(
        f'<div class="chunk-card"><div style="font-family:var(--br-mono);font-size:10px;color:var(--br-text-dim);'
        f'text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px;">Original query</div>'
        f'<div style="font-size:13px;color:var(--br-text);">{escape(trace.query)}</div>'
        + (
            f'<div style="font-family:var(--br-mono);font-size:10px;color:var(--br-text-dim);text-transform:uppercase;'
            f'letter-spacing:.1em;margin:10px 0 5px;">Rewritten query</div>'
            f'<div style="font-family:var(--br-mono);font-size:12px;color:var(--br-text-muted);">{escape(trace.rewritten_query)}</div>'
            if trace.rewritten_query
            else ""
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # --- Execution Waterfall / Step Breakdown ---
    st.markdown("### Pipeline stage waterfall")
    st.caption("Chronological waterfall of the stages actually executed for this request.")

    if not trace.steps:
        empty_state("No sub-steps recorded", "This trace contains only top-level timing information.", mark="—")
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
                badge_color = "var(--br-success)"
                tone = "success"
                status_text = "Completed"
            elif status == "skipped":
                badge_color = "var(--br-text-dim)"
                tone = "default"
                status_text = "Skipped"
            elif status == "warning":
                badge_color = "var(--br-warning)"
                tone = "warning"
                status_text = "Warning"
            else:
                badge_color = "var(--br-danger)"
                tone = "danger"
                status_text = "Failed"

            stage_display_name = step.name.replace("_", " ").title()
            timestamp_short = step.timestamp.split("T")[-1][:12] if "T" in step.timestamp else step.timestamp

            with st.container():
                st.markdown(
                    f'<div class="br-stage" style="border-left-color:{badge_color};">'
                    f'<div class="br-stage-head">'
                    f'<div><span class="br-status-key">Stage {idx}</span> '
                    f'<span class="br-stage-name" style="margin-left:6px;">{escape(stage_display_name)}</span> '
                    f'{pill(status_text, tone)}</div>'
                    f'<div class="br-stage-meta"><span>{escape(timestamp_short)}</span>'
                    f'<span class="br-stage-time">{step.duration_ms:.2f} ms</span></div>'
                    f'</div>'
                    f'<div class="br-track"><span style="margin-left:{step_left_pct:.1f}%;width:{step_width_pct:.1f}%;'
                    f'background:{badge_color};"></span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                with st.expander(f"Inspect stage {idx} payload · {stage_display_name}", expanded=(step.status in ["failed", "warning"])):
                    c_in, c_out = st.columns(2)
                    with c_in:
                        st.markdown("**Input summary**")
                        st.json(step.input_summary if step.input_summary else {"input": "None"})
                    with c_out:
                        st.markdown("**Output summary**")
                        st.json(step.output_summary if step.output_summary else {"output": "None"})
                    if step.error:
                        st.error(f"Execution error: {step.error}")

    st.divider()

    # --- Developer View: Prompt & Context Inspect ---
    st.markdown("### Prompt & retrieved evidence")
    st.caption("Exact context and user prompt passed to the LLM. Secrets and sensitive tokens are redacted before persistence.")

    tab_prompt, tab_ctx, tab_ans = st.tabs(["Full prompt", "Retrieved context", "Model answer"])

    with tab_prompt:
        if trace.prompt_preview:
            st.code(trace.prompt_preview, language="markdown")
        else:
            st.caption("Prompt preview was not logged or was redacted by the active privacy policy.")

    with tab_ctx:
        if trace.context_preview:
            st.code(trace.context_preview, language="markdown")
        else:
            st.caption("Context preview not captured.")

    with tab_ans:
        st.markdown(trace.final_answer or "No answer generated.")
        if trace.citations:
            st.markdown("**Citations attached**")
            for cit in trace.citations:
                st.caption(f"Chapter {cit.get('chapter')} · Page {cit.get('page')}")
