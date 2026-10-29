"""Reader Support & Problem Reporting View.

Allows readers to report issues (wrong answers, hallucination, bad chunks,
upload failures) directly to the engineering team. Tickets are stored in Supabase.
"""

import uuid
import streamlit as st

from observability.models import SupportTicket
from observability.storage import storage
from app.ui import empty_state, escape, page_header, pill


CATEGORIES = [
    "Incorrect Answer / Hallucination",
    "Missing or Broken Citation",
    "Upload or PDF Parsing Failure",
    "Bad or Garbled Text Chunks",
    "Slow / Unresponsive Pipeline",
    "General Question or Feature Request",
]


def render_support_view(pipeline=None):
    """Render the user problem report & support view."""
    page_header(
        "Reader Support · Issue Desk",
        "Report an Issue or Feedback",
        "Encountered a wrong answer, bad chunk, or upload problem? Submit a report directly to the engineering lab.",
    )

    col_form, col_history = st.columns([1.3, 1.2], gap="large")

    with col_form:
        st.markdown(
            """
            <div style="background:var(--br-surface);border:1px solid var(--br-border);border-radius:12px;padding:20px;margin-bottom:20px;">
                <div style="font-size:15px;font-weight:600;color:var(--br-text);margin-bottom:4px;">Submit a Problem Report</div>
                <div style="font-size:13px;color:var(--br-text-muted);">
                    Your feedback is saved into the reliability database so our team can diagnose the root cause using full trace telemetry.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        prefill = st.session_state.pop("prefill_ticket", {}) or {}

        available_books = ["General / All Books"]
        if pipeline:
            try:
                books = pipeline.retrieval_engine.bm25_store.get_distinct_books()
                available_books.extend(books)
            except Exception:
                pass

        cat_idx = 0
        if prefill.get("category") in CATEGORIES:
            cat_idx = CATEGORIES.index(prefill["category"])

        book_idx = 0
        if prefill.get("book_title") and prefill["book_title"] in available_books:
            book_idx = available_books.index(prefill["book_title"])

        with st.form("support_ticket_form", clear_on_submit=True):
            category = st.selectbox("Problem Category", options=CATEGORIES, index=cat_idx)
            book_title = st.selectbox("Related Book (if applicable)", options=available_books, index=book_idx)
            query = st.text_input(
                "Question / Query Asked (optional)",
                value=prefill.get("query", ""),
                placeholder="e.g. What is Constitutional AI?",
            )
            description = st.text_area(
                "Problem Description *",
                value=prefill.get("description", ""),
                placeholder="Describe what went wrong (e.g., The tutor answered with information not found in the book, or page citation was wrong)...",
                height=120,
            )
            contact = st.text_input("Your Name or Email (optional)", placeholder="reader@example.com")

            submitted = st.form_submit_button("📩 Submit Issue to Engineering Lab", type="primary", use_container_width=True)

            if submitted:
                if not description.strip():
                    st.error("Please provide a description of the problem before submitting.")
                else:
                    ticket_id = f"tkt_{uuid.uuid4().hex[:8]}"
                    ticket = SupportTicket(
                        ticket_id=ticket_id,
                        category=category,
                        book_title=book_title if book_title != "General / All Books" else None,
                        query=query.strip() or None,
                        description=description.strip(),
                        user_contact=contact.strip() or None,
                        status="open",
                    )
                    try:
                        storage.save_support_ticket(ticket)
                        st.success(
                            f"**Report submitted successfully!** Ticket ID: `{ticket_id}`. "
                            f"The engineering lab has received your report and can inspect the trace in the Admin Portal."
                        )
                    except Exception as exc:
                        st.error(f"Failed to submit ticket: {exc}")

    with col_history:
        st.markdown(
            """
            <div style="font-size:15px;font-weight:600;color:var(--br-text);margin-bottom:4px;">Recent Community Reports</div>
            <div style="font-size:13px;color:var(--br-text-muted);margin-bottom:16px;">
                Live feed of reported issues tracked in the reliability database:
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            tickets = storage.list_support_tickets(limit=10)
        except Exception:
            tickets = []

        if tickets:
            for t in tickets:
                status_tone = "warning" if t.status == "open" else ("success" if t.status == "resolved" else "neutral")
                time_str = t.timestamp[:16].replace("T", " ") if t.timestamp else ""
                st.markdown(
                    f"""
                    <div style="background:var(--br-surface);border:1px solid var(--br-border);border-radius:10px;padding:14px;margin-bottom:10px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                            <span style="font-family:var(--br-mono);font-size:11px;color:var(--br-text-dim);">{escape(t.ticket_id)} · {escape(time_str)}</span>
                            {pill(t.status.upper(), status_tone)}
                        </div>
                        <div style="font-size:13px;font-weight:600;color:var(--br-text);margin-bottom:4px;">{escape(t.category)}</div>
                        <div style="font-size:12px;color:var(--br-text-muted);margin-bottom:4px;">{escape(t.description[:140])}{'…' if len(t.description) > 140 else ''}</div>
                        {f'<div style="font-size:11px;color:var(--br-primary-light);">📖 {escape(t.book_title)}</div>' if t.book_title else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            empty_state("No reports submitted yet", "Any issues you report will appear here.")
