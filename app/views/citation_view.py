"""Module F: Citation and Answer Verification view."""

import streamlit as st
from typing import List, Dict, Any

from observability.storage import storage
from observability.citation_verifier import citation_verifier
from observability.models import PipelineTrace, CitationAudit


def render_citation_verification():
    """Render Module F - Citation and Answer Verification."""
    st.markdown("## 🛡️ Citation & Answer Verification")
    st.caption("Audit whether generated claims and source citations are strictly grounded in retrieved evidence or constitute ungrounded hallucinations.")

    traces = storage.list_traces(limit=100)
    if not traces:
        st.info("No recorded query traces to audit. Run a question through Book Tutor first.")
        return

    # Select trace to audit
    trace_options = {
        f"{t.trace_id} — '{t.query[:35]}...' ({len(t.citations)} citations)": t
        for t in traces
    }
    selected_trace_key = st.selectbox("Select Answer to Audit Citations:", options=list(trace_options.keys()))
    trace: PipelineTrace = trace_options[selected_trace_key]

    # Reconstruct chunks from trace context or retrieved IDs
    dummy_chunks = []
    for cid in trace.retrieved_chunk_ids:
        # Extract page number if encoded or use metadata
        dummy_chunks.append({"chunk_id": cid, "content": trace.context_preview or "", "metadata": {"page": 1, "chapter": "Chapter 1"}})

    # Run deterministic citation audit
    audit: CitationAudit = citation_verifier.audit_response(
        question=trace.query,
        answer=trace.final_answer,
        retrieved_chunks=dummy_chunks
    )

    # --- Verification Verdict Banner ---
    v_col = "#10b981" if audit.audit_verdict in ("verified", "abstained") else ("#f59e0b" if audit.audit_verdict == "partially_grounded" else "#ef4444")
    st.markdown(
        f'<div style="background-color: {v_col}18; border: 1px solid {v_col}; border-radius: 8px; padding: 14px 18px; margin-bottom: 18px;">'
        f'<div style="font-weight: 700; color: {v_col}; font-size: 1.1rem;">VERDICT: {audit.audit_verdict.upper().replace("_", " ")}</div>'
        f'<div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 4px;">'
        f'Detected {len(audit.detected_citations)} citations. Verified: {len(audit.verified_citations)} | Unsupported: {len(audit.unsupported_citations)} | Invalid: {len(audit.invalid_citations)}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("### 💬 Generated Answer Under Audit")
        st.markdown(trace.final_answer)

        st.divider()
        st.markdown("### 📑 Citation Audit Breakdown")
        if not audit.detected_citations:
            if audit.abstained_properly:
                st.info("✅ Answer correctly abstained with standard 'Not found' response. No citations expected.")
            else:
                st.warning("⚠️ Missing Citations: The answer made assertions without attaching explicit `(Chapter, Page)` citations.")
        else:
            for cit in audit.detected_citations:
                is_invalid = any(c["page"] == cit["page"] for c in audit.invalid_citations)
                is_unsupported = any(c["page"] == cit["page"] for c in audit.unsupported_citations)

                if is_invalid:
                    st.error(f"❌ Invalid Citation: `({cit['chapter']}, Page {cit['page']})` — Page number is outside indexed document range.")
                elif is_unsupported:
                    st.warning(f"⚠️ Unsupported Citation: `({cit['chapter']}, Page {cit['page']})` — Cited source was NOT present in retrieved evidence context.")
                else:
                    st.success(f"✅ Verified Citation: `({cit['chapter']}, Page {cit['page']})` — Grounded in retrieved chunks.")

    with c2:
        st.markdown("### 🔍 Evidence Grounding Diagnostics")
        st.markdown(f"**Should Have Abstained:** `{'Yes' if audit.should_have_abstained else 'No'}`")
        st.markdown(f"**Properly Abstained:** `{'Yes' if audit.abstained_properly else 'No'}`")

        if audit.unsupported_claims:
            st.markdown("#### 🚨 Unsupported Claims Detected")
            for claim in audit.unsupported_claims:
                st.markdown(f"- {claim}")

        st.markdown("#### 📖 Retrieved Context Preview")
        if trace.context_preview:
            st.code(trace.context_preview[:1200] + ("..." if len(trace.context_preview) > 1200 else ""), language="markdown")
        else:
            st.caption("No context recorded.")
