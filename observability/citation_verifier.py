"""Citation and answer verification module with deterministic claim auditing."""

import re
import logging
from typing import List, Dict, Any, Optional

from observability.models import CitationAudit

logger = logging.getLogger(__name__)

# Standard abstention patterns
ABSTENTION_PHRASES = [
    "i couldn't find that in this book",
    "i cannot find that in this book",
    "i could not find sufficient evidence",
    "not mentioned in the provided text",
    "not found in this book",
    "does not contain information about",
]

# Citation regex: (Chapter X, Page Y) or (Chapter ..., Page ...)
CITATION_REGEX = re.compile(
    r"\((?:(?:Chapter|CHAPTER)\s*([^,]+),\s*(?:Page|PAGE)\s*(\d+)|(?:Page|PAGE)\s*(\d+))\)",
    re.IGNORECASE
)


class CitationVerifier:
    """Verifies that citations in generated answers correspond to retrieved evidence."""

    def __init__(self, max_known_page: int = 500):
        self.max_known_page = max_known_page

    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """Extract all (Chapter, Page) citation tuples from text."""
        citations = []
        for match in CITATION_REGEX.finditer(text):
            chap = match.group(1) or "Unknown Chapter"
            p_str = match.group(2) or match.group(3) or "1"
            try:
                page = int(p_str)
            except ValueError:
                page = 1
            citations.append({
                "raw": match.group(0),
                "chapter": chap.strip(),
                "page": page
            })
        return citations

    def audit_response(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        should_abstain: bool = False
    ) -> CitationAudit:
        """
        Audit answer citations against retrieved context chunks.
        Performs deterministic checks on page validity, source existence, and abstention.
        """
        answer_clean = answer.strip().lower()
        abstained = any(phrase in answer_clean for phrase in ABSTENTION_PHRASES)

        # Build map of retrieved sources
        retrieved_sources = set()
        retrieved_pages = set()
        for hit in retrieved_chunks:
            meta = hit.get("metadata", {})
            chap = str(meta.get("chapter", "")).strip().lower()
            p = int(meta.get("page", 0))
            retrieved_sources.add((chap, p))
            if p > 0:
                retrieved_pages.add(p)

        detected = self.extract_citations(answer)
        verified: List[Dict[str, Any]] = []
        invalid: List[Dict[str, Any]] = []
        unsupported: List[Dict[str, Any]] = []
        unsupported_claims: List[str] = []

        for cit in detected:
            page = cit["page"]
            chap_lower = cit["chapter"].lower()

            # Invalid page number check
            if page <= 0 or page > self.max_known_page:
                invalid.append({
                    **cit,
                    "reason": f"Page number {page} is outside valid document bounds (1 to {self.max_known_page})."
                })
                continue

            # Check if this page was retrieved
            if page in retrieved_pages:
                verified.append(cit)
            else:
                unsupported.append({
                    **cit,
                    "reason": f"Page {page} was cited in the answer but was NOT present in retrieved chunks."
                })
                unsupported_claims.append(f"Claim citing ({cit['chapter']}, Page {page}) has no retrieved chunk support.")

        missing_citations = False
        if not abstained and retrieved_chunks and not detected:
            missing_citations = True

        # Determine overall verdict
        if abstained:
            verdict = "abstained"
        elif invalid:
            verdict = "invalid_citations"
        elif unsupported:
            verdict = "unsupported"
        elif missing_citations:
            verdict = "missing_citations"
        elif verified:
            verdict = "verified"
        else:
            verdict = "partially_grounded"

        return CitationAudit(
            answer_text=answer,
            detected_citations=detected,
            verified_citations=verified,
            invalid_citations=invalid,
            unsupported_citations=unsupported,
            unsupported_claims=unsupported_claims,
            missing_citations_flag=missing_citations,
            should_have_abstained=should_abstain or (len(retrieved_chunks) == 0),
            abstained_properly=abstained,
            audit_verdict=verdict
        )


citation_verifier = CitationVerifier()
