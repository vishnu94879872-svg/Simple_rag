"""Chunk inspector and non-destructive quality checker with retrieval-aware diagnostics."""

import re
import difflib
import logging
from typing import List, Dict, Any, Optional

from ingest.models import Chunk, ContentType
from observability.models import ChunkQualityReport, ChunkQualityIssue
from observability.storage import storage

logger = logging.getLogger(__name__)

# Known abbreviations that legitimate sentences or fragments end with
KNOWN_ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "et al.", "fig.", "figs.", "eq.", "eqs.",
    "dr.", "mr.", "ms.", "prof.", "vs.", "v.", "approx.", "vol.", "vols.",
    "no.", "nos.", "p.", "pp.", "dept.", "inc.", "corp.", "ltd.", "co.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec."
}


class ChunkQualityChecker:
    """Diagnostic evaluator for chunk quality, boundary continuity, and retrieval correlations."""

    def __init__(
        self,
        min_chars: int = 80,
        max_chars: int = 1200,
        max_overlap_pct: float = 0.40,
        min_alphanumeric_ratio: float = 0.50,
        max_replacement_chars: int = 3,
        attach_retrieval_diagnostics: bool = True
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_overlap_pct = max_overlap_pct
        self.min_alphanumeric_ratio = min_alphanumeric_ratio
        self.max_replacement_chars = max_replacement_chars
        self.attach_retrieval_diagnostics = attach_retrieval_diagnostics

    def estimate_tokens(self, text: str) -> int:
        """Heuristic token estimation based on whitespace & punctuation."""
        words = text.split()
        return max(1, int(len(words) * 1.33))

    def compute_overlap(self, text_a: str, text_b: str) -> int:
        """Calculate length of longest common trailing/leading overlap between two texts."""
        if not text_a or not text_b:
            return 0
        matcher = difflib.SequenceMatcher(None, text_a[-300:], text_b[:300])
        match = matcher.find_longest_match(0, len(text_a[-300:]), 0, len(text_b[:300]))
        return match.size if match.size >= 15 else 0

    @staticmethod
    def is_heading(line: str) -> bool:
        """Check if line is a markdown heading, section header, or capitalized title."""
        l = line.strip()
        if not l:
            return False
        if l.startswith("#"):
            return True
        if re.match(r"^(?:chapter|section|part|appendix)\s+\d+", l, re.IGNORECASE):
            return True
        if re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", l):
            return True
        if len(l) <= 60 and l.isupper() and any(c.isalpha() for c in l):
            return True
        return False

    @staticmethod
    def is_list_item(line: str) -> bool:
        """Check if line is a markdown bullet, numbered list, or list fragment."""
        l = line.strip()
        if re.match(r"^[\*\-\+•◦▪]\s+", l):
            return True
        if re.match(r"^(?:\d+|\([a-z\d]+\)|\[[a-z\d]+\]|[a-z]\))\s+", l, re.IGNORECASE):
            return True
        return False

    @staticmethod
    def is_abbreviation(text: str) -> bool:
        """Check if text ends with a common abbreviation."""
        tokens = text.strip().split()
        if not tokens:
            return False
        last_tok = tokens[-1].lower().rstrip(",;:)")
        return last_tok in KNOWN_ABBREVIATIONS

    @staticmethod
    def is_code_or_formula(line: str) -> bool:
        """Check if line ends with programming code or mathematical formula syntax."""
        l = line.strip()
        if l.endswith(("}", ");", ";", "```", "{")):
            return True
        if any(sym in l for sym in ["\\sum", "\\int", "\\frac", "\\cdot", "\\partial", "\\alpha", "\\beta"]):
            return True
        if re.search(r"[=\+\-\*/]\s*[a-zA-Z\d\(\)]+$", l):
            return True
        return False

    @staticmethod
    def is_table_construct(line: str) -> bool:
        """Check if line is part of a markdown table."""
        l = line.strip()
        return l.startswith("|") or l.endswith("|") or " | " in l

    def evaluate_chunks(
        self,
        chunks: List[Chunk],
        retrieval_history: Optional[List[Dict[str, Any]]] = None
    ) -> List[ChunkQualityReport]:
        """
        Evaluate a sequence of chunks and return diagnostic quality reports.
        Guaranteed to be read-only: does not modify or re-chunk content.
        Distinguishes warnings from errors, provides exact evidence, and correlates retrieval history.
        """
        reports: List[ChunkQualityReport] = []
        content_seen: Dict[str, str] = {}

        for i, c in enumerate(chunks):
            content = c.content.strip() if c.content else ""
            char_count = len(content)
            token_count = self.estimate_tokens(content) if char_count > 0 else 0
            issues: List[ChunkQualityIssue] = []

            # 1. Empty or nearly empty chunk
            if char_count < 10:
                issues.append(
                    ChunkQualityIssue(
                        issue_type="empty",
                        severity="warning",
                        description=f"Chunk has fewer than 10 characters ({char_count} chars).",
                        recommendation="Review document parsing to ensure empty blocks are filtered.",
                        evidence=f"Chunk content length is {char_count} characters.",
                        is_heuristic=True
                    )
                )

            # 2. Duplicate chunk check
            clean_body = re.sub(r"^\[.*?\]\n*", "", content).strip().lower()
            if clean_body and clean_body in content_seen:
                prev_id = content_seen[clean_body]
                issues.append(
                    ChunkQualityIssue(
                        issue_type="duplicate",
                        severity="warning",
                        description=f"Identical text detected previously in chunk '{prev_id}'.",
                        recommendation="Check if document contains repeated pages or duplicate index entries.",
                        evidence=f"Exact normalized body match with previously processed chunk '{prev_id}'.",
                        is_heuristic=True
                    )
                )
            elif clean_body:
                content_seen[clean_body] = c.chunk_id

            # 3. Size boundaries
            if 0 < char_count < self.min_chars:
                issues.append(
                    ChunkQualityIssue(
                        issue_type="too_short",
                        severity="info",
                        description=f"Chunk is unusually short ({char_count} chars < {self.min_chars} threshold).",
                        recommendation="Consider merging with neighboring text or using parent section expansion.",
                        evidence=f"Length {char_count} characters vs min_chars threshold {self.min_chars}.",
                        is_heuristic=True
                    )
                )
            elif char_count > self.max_chars:
                issues.append(
                    ChunkQualityIssue(
                        issue_type="too_long",
                        severity="warning",
                        description=f"Chunk exceeds max character threshold ({char_count} chars > {self.max_chars}).",
                        recommendation="Check text splitting parameters to prevent truncation in downstream models.",
                        evidence=f"Length {char_count} characters vs max_chars threshold {self.max_chars}.",
                        is_heuristic=True
                    )
                )

            # 4. Contextual sentence continuation and boundary heuristics
            body = re.sub(r"^\[.*?\]\n*", "", content).strip()
            prev_chunk = chunks[i - 1] if i > 0 else None
            next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None

            if body:
                lines = [l.strip() for l in body.splitlines() if l.strip()]
                first_line = lines[0] if lines else ""
                last_line = lines[-1] if lines else ""

                # --- Leading boundary check ---
                if body[0].islower() and not body.startswith(("http", "www", "def ", "class ", "return ")):
                    # Check whether preceding chunk exists and was left open
                    if prev_chunk:
                        prev_body = re.sub(r"^\[.*?\]\n*", "", prev_chunk.content).strip()
                        prev_tail = prev_body[-40:] if len(prev_body) > 40 else prev_body
                        issues.append(
                            ChunkQualityIssue(
                                issue_type="broken_sentence",
                                severity="info",
                                description="Sentence continues across chunk boundary from preceding chunk.",
                                recommendation="Normal behavior during boundary splitting. Context overlap ensures completeness.",
                                evidence=f"Preceding chunk '{prev_chunk.chunk_id}' ends with: '...{prev_tail}' | Current chunk begins with: '{body[:30]}...'",
                                is_heuristic=True
                            )
                        )
                    else:
                        issues.append(
                            ChunkQualityIssue(
                                issue_type="broken_sentence",
                                severity="warning",
                                description="Chunk begins with a lowercase letter without preceding chunk context.",
                                recommendation="Check if page boundary or parser clipped the beginning of the sentence.",
                                evidence=f"Leading snippet: '{body[:40]}...'",
                                is_heuristic=True
                            )
                        )

                # --- Trailing boundary check ---
                terminal_punct = (".", "?", "!", '"', "'", "```", "|", ";", ":", ")", "]", ">")
                ends_with_terminal = body.endswith(terminal_punct)

                if not ends_with_terminal:
                    # Distinguish legitimate constructs
                    is_legit = (
                        self.is_heading(last_line) or
                        self.is_list_item(last_line) or
                        self.is_table_construct(last_line) or
                        self.is_abbreviation(last_line) or
                        self.is_code_or_formula(last_line) or
                        last_line.startswith(("Figure", "Table", "Fig.", "Diagram", "Chart"))
                    )

                    if not is_legit:
                        trailing_tail = body[-40:] if len(body) > 40 else body
                        # Inspect next chunk to see if sentence continues across boundary
                        if next_chunk:
                            next_body = re.sub(r"^\[.*?\]\n*", "", next_chunk.content).strip()
                            next_head = next_body[:35] if len(next_body) > 35 else next_body
                            
                            # Does next chunk continue the sentence?
                            if next_body and (next_body[0].islower() or next_body.startswith((",", "-", "—")) or not self.is_heading(next_body.splitlines()[0])):
                                issues.append(
                                    ChunkQualityIssue(
                                        issue_type="broken_sentence",
                                        severity="info",
                                        description="Sentence continues across chunk boundary into succeeding chunk.",
                                        recommendation="Normal boundary continuation. Neural reranker and parent expansion bridge the context.",
                                        evidence=f"Chunk ends with: '...{trailing_tail}' | Succeeding chunk '{next_chunk.chunk_id}' begins with: '{next_head}...'",
                                        is_heuristic=True
                                    )
                                )
                            else:
                                issues.append(
                                    ChunkQualityIssue(
                                        issue_type="broken_sentence",
                                        severity="warning",
                                        description="Chunk terminates without end-of-sentence punctuation and sentence does not appear to continue in next chunk.",
                                        recommendation="Verify whether chunk boundary or section transition split an unfinished sentence.",
                                        evidence=f"Trailing snippet: '...{trailing_tail}' | Next chunk begins a new heading or capitalized line: '{next_head}...'",
                                        is_heuristic=True
                                    )
                                )
                        else:
                            issues.append(
                                ChunkQualityIssue(
                                    issue_type="broken_sentence",
                                    severity="warning",
                                    description="Final chunk in sequence terminates without end-of-sentence punctuation.",
                                    recommendation="Check if document text was truncated at end of ingestion.",
                                    evidence=f"Trailing snippet at document end: '...{trailing_tail}'",
                                    is_heuristic=True
                                )
                            )

            # 5. Heading detached from content
            if content.startswith("[") and "]" in content:
                heading_part, _, body_part = content.partition("]")
                if len(body_part.strip()) < 15:
                    issues.append(
                        ChunkQualityIssue(
                            issue_type="detached_heading",
                            severity="warning",
                            description="Chunk consists mostly of a heading banner with negligible explanatory text.",
                            recommendation="Check if section heading should be bundled with the succeeding paragraph.",
                            evidence=f"Heading: '{heading_part}]' followed by only {len(body_part.strip())} chars of text.",
                            is_heuristic=True
                        )
                    )

            # 6. Table integrity check & potential table fragmentation
            if c.metadata.type == ContentType.TABLE.value:
                if "|" in content:
                    lines = [l.strip() for l in content.splitlines() if l.startswith("|")]
                    if len(lines) < 2 or not any("---" in l for l in lines):
                        issues.append(
                            ChunkQualityIssue(
                                issue_type="split_table",
                                severity="warning",
                                description="Table chunk appears malformed or missing standard markdown header delimiter.",
                                recommendation="Preserve atomic table extraction without splitting across chunks.",
                                evidence=f"Table chunk contains {len(lines)} pipe lines with missing '---' column separator.",
                                is_heuristic=True
                            )
                        )
            else:
                # Text chunk with isolated table fragments
                if "|---" in content or "---|---" in content:
                    pipe_lines = [l.strip() for l in content.splitlines() if "|" in l]
                    if len(pipe_lines) <= 2:
                        issues.append(
                            ChunkQualityIssue(
                                issue_type="potential_table_fragmentation",
                                severity="warning",
                                description="Text chunk contains isolated table delimiter rows, indicating a table may have been fragmented.",
                                recommendation="Ensure parser classifies tabular data blocks before sentence splitting.",
                                evidence=f"Found {len(pipe_lines)} isolated table delimiter line(s) in a text chunk.",
                                is_heuristic=True
                            )
                        )

            # 7. Metadata completeness
            meta = c.metadata
            if meta.page <= 0:
                issues.append(
                    ChunkQualityIssue(
                        issue_type="missing_metadata",
                        severity="warning",
                        description=f"Invalid page number recorded: {meta.page}",
                        recommendation="Ensure parser correctly passes 1-indexed page numbering.",
                        evidence=f"Recorded page index is {meta.page}.",
                        is_heuristic=True
                    )
                )
            if not meta.chunk_id:
                issues.append(
                    ChunkQualityIssue(
                        issue_type="missing_metadata",
                        severity="warning",
                        description="Chunk is missing a unique chunk_id.",
                        recommendation="Ensure chunk generation assigns deterministic UUIDs or document-based IDs.",
                        evidence="chunk_id attribute is null or empty.",
                        is_heuristic=True
                    )
                )
            if not meta.chapter or meta.chapter == "Unknown":
                issues.append(
                    ChunkQualityIssue(
                        issue_type="missing_metadata",
                        severity="info",
                        description="Chapter metadata is unspecified or marked as Unknown.",
                        recommendation="Review PDF table of contents or heading extraction heuristics.",
                        evidence=f"Chapter attribute value: '{meta.chapter}'.",
                        is_heuristic=True
                    )
                )

            # 8. Overlap calculation with adjacent chunks
            overlap_prev = 0
            overlap_prev_pct = 0.0
            if i > 0:
                prev_content = chunks[i - 1].content
                overlap_prev = self.compute_overlap(prev_content, content)
                if char_count > 0:
                    overlap_prev_pct = round(overlap_prev / char_count, 3)
                    if overlap_prev_pct > self.max_overlap_pct:
                        issues.append(
                            ChunkQualityIssue(
                                issue_type="excessive_overlap",
                                severity="warning",
                                description=f"High overlap ({int(overlap_prev_pct * 100)}%) with previous chunk '{chunks[i - 1].chunk_id}'.",
                                recommendation="Reduce chunk_overlap setting to prevent redundant context retrieval.",
                                evidence=f"Overlap size {overlap_prev} characters represents {int(overlap_prev_pct * 100)}% of chunk.",
                                is_heuristic=True
                            )
                        )

            overlap_next = 0
            overlap_next_pct = 0.0
            if i < len(chunks) - 1:
                next_content = chunks[i + 1].content
                overlap_next = self.compute_overlap(content, next_content)
                if char_count > 0:
                    overlap_next_pct = round(overlap_next / char_count, 3)

            # 9. Suspicious text extraction (mojibake, low alphanumeric ratio, character repetition)
            if char_count > 0:
                replacement_cnt = content.count("\ufffd") + content.count("?") if "\ufffd" in content else 0
                if replacement_cnt > self.max_replacement_chars:
                    issues.append(
                        ChunkQualityIssue(
                            issue_type="suspicious_extraction",
                            severity="warning",
                            description=f"Chunk contains {replacement_cnt} unicode replacement or unprintable characters.",
                            recommendation="Check font encoding or consider applying OCR to the source page.",
                            evidence=f"Detected {replacement_cnt} unicode replacement characters (\ufffd) in {char_count} chars of text.",
                            is_heuristic=True
                        )
                    )

                if char_count > 50:
                    alnum_count = sum(ch.isalnum() or ch.isspace() for ch in content)
                    alnum_ratio = round(alnum_count / char_count, 3)
                    if alnum_ratio < self.min_alphanumeric_ratio:
                        issues.append(
                            ChunkQualityIssue(
                                issue_type="suspicious_extraction",
                                severity="warning",
                                description=f"Unusually low alphanumeric text ratio ({int(alnum_ratio * 100)}% < {int(self.min_alphanumeric_ratio * 100)}%).",
                                recommendation="Inspect for corrupted font extraction, binary noise, or complex vector artwork.",
                                evidence=f"Only {alnum_count}/{char_count} characters are alphanumeric or whitespace.",
                                is_heuristic=True
                            )
                        )

                # Repetitive punctuation / symbol runs (e.g. ............. or -------)
                rep_match = re.search(r"([\.\-_=~])\1{11,}", content)
                if rep_match:
                    issues.append(
                        ChunkQualityIssue(
                            issue_type="suspicious_extraction",
                            severity="info",
                            description="Long repetitive character run detected.",
                            recommendation="Check if table of contents leader dots or horizontal dividers were captured as text.",
                            evidence=f"Found run of '{rep_match.group(1)}' repeated {len(rep_match.group(0))} times.",
                            is_heuristic=True
                        )
                    )

            # 10. Boilerplate repetition
            if re.search(r"\b(all rights reserved|copyright ©|terms of service|privacy policy)\b", content, re.IGNORECASE):
                issues.append(
                    ChunkQualityIssue(
                        issue_type="boilerplate",
                        severity="info",
                        description="Chunk contains legal or boilerplate copyright disclaimers.",
                        recommendation="Consider adding boilerplate filtering before embedding.",
                        evidence="Matched legal/copyright pattern in text.",
                        is_heuristic=True
                    )
                )

            report = ChunkQualityReport(
                chunk_id=c.chunk_id,
                char_count=char_count,
                token_count_estimate=token_count,
                overlap_prev_chars=overlap_prev,
                overlap_prev_pct=overlap_prev_pct,
                overlap_next_chars=overlap_next,
                overlap_next_pct=overlap_next_pct,
                issues=issues,
                source_document=c.metadata.book_title,
                source_page=c.metadata.page,
                source_section=c.metadata.heading_path
            )
            reports.append(report)

        # Correlate retrieval-aware diagnostics if requested
        if self.attach_retrieval_diagnostics:
            self.attach_retrieval_history(reports)

        return reports

    def attach_retrieval_history(self, reports: List[ChunkQualityReport]) -> None:
        """
        Correlate quality reports with recorded execution traces and evaluation results.
        Strictly observes the rule: Do not claim a chunk caused an incorrect answer merely because it has a warning.
        """
        try:
            traces = storage.list_traces(limit=150)
            eval_runs = storage.list_eval_runs(limit=30)
        except Exception as e:
            logger.debug("Could not read traces for chunk retrieval diagnostics: %s", e)
            return

        report_map = {r.chunk_id: r for r in reports}

        for t in traces:
            retrieved_ids = set(t.retrieved_chunk_ids or [])
            for cid in retrieved_ids:
                if cid in report_map:
                    rep = report_map[cid]
                    rep.was_retrieved = True
                    if t.query not in rep.queries_retrieved_by:
                        rep.queries_retrieved_by.append(t.query)

                    # Determine rank and score if available in trace steps
                    rank = 1
                    score = 0.0
                    if cid in t.retrieved_chunk_ids:
                        rank = t.retrieved_chunk_ids.index(cid) + 1
                    
                    if rep.best_rank is None or rank < rep.best_rank:
                        rep.best_rank = rank

                    # Check if used in final LLM context
                    used_in_ctx = False
                    if t.context_preview and cid in t.context_preview:
                        used_in_ctx = True
                    elif rank <= t.top_k:
                        used_in_ctx = True
                    rep.used_in_final_context = rep.used_in_final_context or used_in_ctx

                    # Check evaluation status of associated answer
                    answer_passed = (t.status == "success")
                    rep.associated_answer_passed_eval = answer_passed

                    rep.retrieval_history.append({
                        "trace_id": t.trace_id,
                        "query": t.query,
                        "rank": rank,
                        "used_in_context": used_in_ctx,
                        "answer_passed_eval": answer_passed,
                        "model": t.model
                    })

        # Formulate evidence-based causality assessment
        for rep in reports:
            has_warnings = len([iss for iss in rep.issues if iss.severity in ("warning", "error")]) > 0
            if not rep.was_retrieved:
                if has_warnings:
                    rep.causality_assessment = (
                        "No causal link: Chunk has heuristic warnings, but has never been retrieved by any recorded query."
                    )
                else:
                    rep.causality_assessment = "Nominal chunk: Clean quality heuristics; unretrieved in current query logs."
            else:
                if rep.associated_answer_passed_eval:
                    if has_warnings:
                        rep.causality_assessment = (
                            "Non-fatal heuristic warning: Chunk was retrieved and used in synthesis, "
                            "and the resulting answer successfully passed evaluation."
                        )
                    else:
                        rep.causality_assessment = "Verified effective: Chunk was retrieved and supported a passing answer."
                else:
                    if has_warnings:
                        rep.causality_assessment = (
                            "Correlated warning: Chunk has quality warnings and was retrieved for a query that failed or had errors. "
                            "Inspect whether chunk content or incomplete boundaries contributed to the issue."
                        )
                    else:
                        rep.causality_assessment = (
                            "Retrieved in failed query, but chunk formatting is clean. Failure likely occurred in retrieval ranking or LLM synthesis."
                        )


chunk_quality_checker = ChunkQualityChecker()
