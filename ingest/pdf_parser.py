"""PDF parsing module with layout, block-type detection, table extraction, and image extraction."""

import io
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pymupdf
from PIL import Image

from config import settings
from ingest.models import ParsedBlock, ContentType
from ingest.keyword_extractor import extract_keywords_from_text

logger = logging.getLogger(__name__)

# Monospace font indicators
MONOSPACE_FONTS = {"courier", "consolas", "monaco", "menlo", "dejavusansmono", "sourcecodepro", "monospace"}

# Bullet markers regex
LIST_ITEM_REGEX = re.compile(r"^\s*(?:[\u2022\u25E6\u2043\u2219\*\-–—]|\d+[\.\)]|[a-zA-Z][\.\)])\s+", re.MULTILINE)

# Strict code syntax regex patterns (requires real programming constructs, not isolated English words)
STRICT_CODE_PATTERNS = [
    r"^\s*(def|async\s+def)\s+\w+\s*\(",
    r"^\s*function\s+\w+\s*\(",
    r"^\s*(const|let|var)\s+\w+\s*=",
    r"^\s*(import|from)\s+[a-zA-Z0-9_\.]+\s+(import|as)\b",
    r"^\s*class\s+[A-Z]\w*(?:\([^)]*\))?:",
    r"^\s*(public|private|protected)\s+(?:static\s+)?(?:void|int|string|boolean|float)\b",
    r"^\s*#include\s+<[^>]+>",
    r"^\s*for\s*\([^;]+;[^;]+;[^)]+\)",
    r"^\s*(return|yield)\s+[a-zA-Z0-9_\(\{\[\"'].*[;]?$",
    r"[{};]\s*$"
]
CODE_REGEX = re.compile("|".join(STRICT_CODE_PATTERNS))


class HeadingTracker:
    """Maintains chapter, section, and subsection hierarchy across pages."""

    def __init__(self, toc: List[List[Any]]):
        # toc item: [level, title, page_number] (1-indexed page_number)
        self.toc = toc
        self.chapter: str = "Introduction"
        self.section: str = "General"
        self.subsection: str = ""

    def update_from_toc(self, page_num: int) -> None:
        """Update hierarchy based on PDF TOC entries matching current page."""
        for level, title, p_num in self.toc:
            if p_num == page_num:
                clean_title = title.strip()
                if level == 1:
                    self.chapter = clean_title
                    self.section = "Overview"
                    self.subsection = ""
                elif level == 2:
                    self.section = clean_title
                    self.subsection = ""
                elif level >= 3:
                    self.subsection = clean_title

    def update_from_text(self, text: str) -> None:
        """Heuristically update headings if text indicates a clear Chapter/Section header."""
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if not lines:
            return
        first_line = lines[0]
        # 1. Check for Chapter N or CHAPTER N patterns
        chap_match = re.match(r"^(?:Chapter|CHAPTER)\s+([0-9IVXLCDM]+)[:\.\s]*(.*)", first_line, re.IGNORECASE)
        if chap_match:
            title = chap_match.group(2).strip() or f"Chapter {chap_match.group(1)}"
            self.chapter = f"Chapter {chap_match.group(1)}: {title}" if title else f"Chapter {chap_match.group(1)}"
            self.section = "Overview"
            self.subsection = ""
            return

        # 2. Check for numbered major headers like "1. Amphibians", "2. Mammals", "1. Introduction"
        num_major_match = re.match(r"^(\d+)\.\s+([A-Z][A-Za-z0-9\s\-_,]{2,60})$", first_line)
        if num_major_match and len(lines) <= 2:
            num = num_major_match.group(1)
            title = num_major_match.group(2).strip()
            self.chapter = f"Chapter {num}: {title}"
            self.section = "Overview"
            self.subsection = ""
            return

        # 3. Check for Section N or N.N (e.g. 1.2 Introduction)
        sec_match = re.match(r"^(?:Section|\d+\.\d+)\s*[:\.\s]*(.*)", first_line, re.IGNORECASE)
        if sec_match and len(first_line) < 80:
            self.section = first_line
            self.subsection = ""
            return

        # 4. Short stand-alone section titles (e.g. "Defining features", "Examples and habitats")
        if len(lines) == 1 and 3 <= len(first_line) <= 45 and not first_line.endswith(('.', ':', '?', '!')):
            if first_line[0].isupper() and not any(ch in first_line for ch in ';={}()|'):
                if self.section != "Overview":
                    self.subsection = first_line
                else:
                    self.section = first_line

    @property
    def heading_path(self) -> str:
        parts = [self.chapter]
        if self.section and self.section != "Overview":
            parts.append(self.section)
        if self.subsection:
            parts.append(self.subsection)
        return " > ".join(parts)


def is_header_or_footer(bbox: pymupdf.Rect, page_rect: pymupdf.Rect, text: str) -> bool:
    """Identify running headers, footers, and standalone page numbers to avoid polluting chunk content."""
    clean = text.strip()
    if not clean:
        return True

    # Standalone page number e.g. "12", "Page 3", "4 of 10"
    if re.match(r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$", clean, re.IGNORECASE):
        if bbox.y1 <= 65 or bbox.y0 >= (page_rect.height - 65):
            return True

    # Extreme top strip (< 48pt) or extreme bottom strip (> page_height - 48pt)
    in_top_strip = bbox.y1 <= 48
    in_bottom_strip = bbox.y0 >= (page_rect.height - 48)
    if in_top_strip or in_bottom_strip:
        # Common running header / footer patterns
        if re.search(r"(?:\|\s*page|\bpage\s+\d+\b|document id:|all rights reserved|confidential)", clean, re.IGNORECASE):
            return True
        if len(clean) < 90 and not clean.endswith(('.', '?', '!')):
            return True

    # Document ID lines anywhere on the page
    if re.match(r"^document id:\s*[a-z0-9\-]+", clean, re.IGNORECASE):
        return True

    return False


def detect_block_type(text: str, is_monospace: bool = False) -> ContentType:
    """
    Detect the content type of a text block:
    text | list | code
    (Tables and images are detected separately via structural extractors).
    """
    text_stripped = text.strip()
    if not text_stripped:
        return ContentType.TEXT

    # 1. Code detection: markdown fences
    if text_stripped.startswith("```") or text_stripped.endswith("```"):
        return ContentType.CODE

    # 2. Monospace font with genuine code syntax characters
    if is_monospace:
        if any(sym in text_stripped for sym in ["{", "}", ";", "(", ")", "=", "->", "=>"]):
            return ContentType.CODE

    lines = [l for l in text_stripped.splitlines() if l.strip()]
    if lines:
        # Check strict programming code patterns
        code_matches = sum(1 for line in lines if CODE_REGEX.search(line))
        if len(lines) >= 2 and (code_matches / len(lines)) >= 0.5:
            return ContentType.CODE

    # 3. List detection
    bullet_lines = sum(1 for line in lines if LIST_ITEM_REGEX.match(line))
    if len(lines) > 1 and (bullet_lines / len(lines)) >= 0.6:
        return ContentType.LIST

    return ContentType.TEXT


def sanitize_filename(name: str) -> str:
    """Convert string to safe filename."""
    return re.sub(r"[^\w\-_\.]", "_", name)


class PDFParser:
    """Extracts structured blocks (text, tables, images) preserving hierarchy."""

    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        self.doc = pymupdf.open(str(self.pdf_path))
        self.book_title = self._extract_title()
        self.toc = self.doc.get_toc()  # [[lvl, title, page], ...]
        self.heading_tracker = HeadingTracker(self.toc)

    def _extract_title(self) -> str:
        """Extract book title from PDF metadata or filename."""
        meta_title = self.doc.metadata.get("title")
        if meta_title and meta_title.strip():
            return meta_title.strip()
        # Fallback to stem
        return self.pdf_path.stem.replace("_", " ").title()

    def parse(self) -> List[ParsedBlock]:
        """Parse the entire document into structured ParsedBlock objects."""
        all_blocks: List[ParsedBlock] = []

        for page_idx in range(len(self.doc)):
            page_num = page_idx + 1
            page = self.doc[page_idx]
            self.heading_tracker.update_from_toc(page_num)

            # 1. Extract tables first so we can obtain bounding boxes
            table_blocks, table_bboxes = self._extract_tables(page, page_num)
            all_blocks.extend(table_blocks)

            # 2. Extract images
            image_blocks = self._extract_images(page, page_num)
            all_blocks.extend(image_blocks)

            # 3. Extract text / list / code blocks, skipping table bounding boxes
            text_blocks = self._extract_text_blocks(page, page_num, table_bboxes)
            all_blocks.extend(text_blocks)

        logger.info(
            "Parsed %d total blocks from '%s' (%d pages)",
            len(all_blocks),
            self.book_title,
            len(self.doc)
        )
        return all_blocks

    def _extract_tables(self, page: pymupdf.Page, page_num: int) -> Tuple[List[ParsedBlock], List[pymupdf.Rect]]:
        """Extract tables from page and convert them to markdown."""
        blocks: List[ParsedBlock] = []
        bboxes: List[pymupdf.Rect] = []

        try:
            tabs = page.find_tables()
            for tab in tabs:
                df = tab.extract()
                if not df or len(df) < 2:
                    continue  # Skip trivial/empty tables

                # Convert table rows to markdown format
                headers = [str(c or "").strip() for c in df[0]]
                sep = ["---"] * len(headers)
                md_lines = [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(sep) + " |"
                ]
                for row in df[1:]:
                    row_strs = [str(c or "").strip().replace("\n", " ") for c in row]
                    # Ensure matching length
                    while len(row_strs) < len(headers):
                        row_strs.append("")
                    md_lines.append("| " + " | ".join(row_strs[:len(headers)]) + " |")

                table_md = "\n".join(md_lines)
                keywords = extract_keywords_from_text(table_md)

                block = ParsedBlock(
                    content_type=ContentType.TABLE,
                    content=table_md,
                    page=page_num,
                    chapter=self.heading_tracker.chapter,
                    section=self.heading_tracker.section,
                    heading_path=self.heading_tracker.heading_path,
                    keywords=keywords,
                    table_markdown=table_md
                )
                blocks.append(block)
                bboxes.append(tab.bbox)
        except Exception as e:
            logger.warning("Table extraction error on page %d: %s", page_num, e)

        return blocks, bboxes

    def _extract_images(self, page: pymupdf.Page, page_num: int) -> List[ParsedBlock]:
        """Extract images from page and save to data/images."""
        blocks: List[ParsedBlock] = []
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = self.doc.extract_image(xref)
                if not base_image:
                    continue
                image_bytes = base_image["image"]
                image_ext = base_image.get("ext", "png")

                # Filter out small decorative icons / stamps (< 60px)
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                if width < 60 or height < 60:
                    continue

                safe_title = sanitize_filename(self.book_title)
                img_filename = f"{safe_title}_p{page_num}_img{img_idx + 1}.{image_ext}"
                img_path = settings.images_dir / img_filename

                # Save image file
                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                block = ParsedBlock(
                    content_type=ContentType.IMAGE,
                    content="",  # Caption will be generated by LLM helper
                    page=page_num,
                    chapter=self.heading_tracker.chapter,
                    section=self.heading_tracker.section,
                    heading_path=self.heading_tracker.heading_path,
                    image_path=str(img_path),
                    keywords=[f"Figure {img_idx + 1}", "Image", f"Page {page_num}"]
                )
                blocks.append(block)
            except Exception as e:
                logger.warning("Image extraction error on page %d (xref %d): %s", page_num, xref, e)

        return blocks

    def _extract_text_blocks(
        self,
        page: pymupdf.Page,
        page_num: int,
        table_bboxes: List[pymupdf.Rect]
    ) -> List[ParsedBlock]:
        """Extract text blocks with font/style inspection, avoiding table regions and running headers/footers."""
        blocks: List[ParsedBlock] = []
        # sort=True ensures blocks follow natural reading order (top-to-bottom, columns)
        page_dict = page.get_text("dict", sort=True)

        for b in page_dict.get("blocks", []):
            # Check if block is an image or outside text
            if b.get("type") != 0:  # 0 is text block
                continue

            bbox = pymupdf.Rect(b["bbox"])
            # Check if inside any table bounding box
            is_inside_table = any(bbox.intersects(tab_box) for tab_box in table_bboxes)
            if is_inside_table:
                continue

            block_text_lines: List[str] = []
            bold_spans: List[str] = []
            is_monospace = False

            for line in b.get("lines", []):
                line_spans: List[str] = []
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if not span_text:
                        continue
                    font = span.get("font", "").lower()
                    flags = span.get("flags", 0)

                    # Check bold
                    if (flags & 2 != 0) or ("bold" in font) or ("black" in font):
                        if span_text.strip():
                            bold_spans.append(span_text.strip())

                    # Check monospace font
                    if any(m in font for m in MONOSPACE_FONTS):
                        is_monospace = True

                    # Word-spacing preservation:
                    # If previous span does not end with space and current span does not start with space,
                    # insert space so styled words do not concatenate together.
                    if line_spans and not line_spans[-1].endswith(" ") and not span_text.startswith(" "):
                        line_spans.append(" ")

                    line_spans.append(span_text)

                if line_spans:
                    line_str = "".join(line_spans).strip()
                    if line_str:
                        block_text_lines.append(line_str)

            if not block_text_lines:
                continue

            # Join lines inside block:
            # 1. De-hyphenate words broken across lines (e.g. "experi-\nment" -> "experiment")
            raw_text = "\n".join(block_text_lines)
            cleaned_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", raw_text)

            # 2. If standard prose (not monospace or list), join soft-wrapped lines into coherent sentences
            if not is_monospace and not any(LIST_ITEM_REGEX.match(l) for l in block_text_lines):
                cleaned_text = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned_text)

            full_block_text = cleaned_text.strip()
            if not full_block_text:
                continue

            # Filter out running headers, footers, and standalone page numbers
            if is_header_or_footer(bbox, page.rect, full_block_text):
                continue

            # Update heading hierarchy if block is a heading
            self.heading_tracker.update_from_text(full_block_text)

            content_type = detect_block_type(full_block_text, is_monospace=is_monospace)
            keywords = extract_keywords_from_text(full_block_text, additional_terms=bold_spans)

            block = ParsedBlock(
                content_type=content_type,
                content=full_block_text,
                page=page_num,
                chapter=self.heading_tracker.chapter,
                section=self.heading_tracker.section,
                heading_path=self.heading_tracker.heading_path,
                keywords=keywords
            )
            blocks.append(block)

        return blocks
