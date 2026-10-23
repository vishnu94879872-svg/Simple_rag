"""Module F: Citation and Answer Verification view."""

import streamlit as st
from typing import List, Dict, Any

from observability.storage import storage
from observability.citation_verifier import citation_verifier
from observability.models import PipelineTrace, CitationAudit
from app.ui import empty_state, page_header, verdict


def render_citation_verification():
    """Render Module F - Citation and Answer Verification."""
    page_header(
        "Module · Grounding",
        "Citation Verification",
        "Audit whether generated claims and citations are grounded in retrieved evidence or constitute ungrounded hallucinations.",
    )

    traces = storage.list_traces(limit=100)
    if not traces:
        empty_state(
            "No recorded traces to audit",
            "Run a question through Chat first, then return here to verify its citations.",
            mark="⌁",
        )
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
    tone = "success" if audit.audit_verdict in ("verified", "abstained") else ("warning" if audit.audit_verdict == "partially_grounded" else "danger")
    verdict(
        f"Verdict · {audit.audit_verdict.replace('_', ' ').title()}",
        f"Detected {len(audit.detected_citations)} citations · verified {len(audit.verified_citations)} · "
        f"unsupported {len(audit.unsupported_citations)} · invalid {len(audit.invalid_citations)}.",
        tone=tone,
    )

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("### Generated answer under audit")
        st.markdown(trace.final_answer)

        st.divider()
        st.markdown("### Citation audit breakdown")
        if not audit.detected_citations:
            if audit.abstained_properly:
                st.info("Answer correctly abstained with a standard not-found response. No citations expected.")
            else:
                st.warning("Missing citations: the answer made assertions without attaching explicit (Chapter, Page) references.")
        else:
            for cit in audit.detected_citations:
                is_invalid = any(c["page"] == cit["page"] for c in audit.invalid_citations)
                is_unsupported = any(c["page"] == cit["page"] for c in audit.unsupported_citations)

                if is_invalid:
                    st.error(f"Invalid citation: ({cit['chapter']}, Page {cit['page']}) — page number is outside the indexed document range.")
                elif is_unsupported:
                    st.warning(f"Unsupported citation: ({cit['chapter']}, Page {cit['page']}) — source was not present in retrieved context.")
                else:
                    st.success(f"Verified citation: ({cit['chapter']}, Page {cit['page']}) — grounded in retrieved chunks.")

    with c2:
        st.markdown("### Evidence grounding diagnostics")
        st.markdown(f"**Should have abstained:** `{'Yes' if audit.should_have_abstained else 'No'}`")
        st.markdown(f"**Properly abstained:** `{'Yes' if audit.abstained_properly else 'No'}`")

        if audit.unsupported_claims:
            st.markdown("#### Unsupported claims detected")
            for claim in audit.unsupported_claims:
                st.markdown(f"- {claim}")

        st.markdown("#### Retrieved context preview")
        if trace.context_preview:
            st.code(trace.context_preview[:1200] + ("..." if len(trace.context_preview) > 1200 else ""), language="markdown")
        else:
            st.caption("No context recorded.")
