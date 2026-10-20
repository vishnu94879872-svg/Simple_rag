"""Text splitter and chunk constructor enforcing heading prefixes and sentence integrity."""

import hashlib
import logging
from typing import List, Dict, Tuple, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from ingest.models import ParsedBlock, Chunk, ChunkMetadata, ParentSection, ContentType
from ingest.llm_helper import LLMHelper
from ingest.keyword_extractor import extract_keywords_from_text

logger = logging.getLogger(__name__)


def generate_id(prefix: str, content: str) -> str:
    """Create a short deterministic hash ID."""
    digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


class ChunkBuilder:
    """Transforms parsed blocks into validated chunks and parent sections."""

    def __init__(self, book_title: str, llm_helper: LLMHelper = None):
        self.book_title = book_title
        self.llm_helper = llm_helper or LLMHelper()
        # Sentence-preserving splitter with character-level overflow guard
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
            keep_separator=True,
            is_separator_regex=False
        )

    def _sanitize_blocks(self, blocks: Any) -> List[ParsedBlock]:
        """Validate and sanitize parsed blocks at the parsing-to-chunking boundary."""
        if not blocks:
            return []
        sanitized: List[ParsedBlock] = []
        for i, b in enumerate(blocks):
            if b is None:
                continue
            if isinstance(b, dict):
                try:
                    b = ParsedBlock(**b)
                except Exception as e:
                    logger.warning("Skipping invalid dict block at index %d: %s", i, e)
                    continue
            elif not isinstance(b, ParsedBlock):
                logger.warning("Skipping block of unrecognized type %s at index %d", type(b), i)
                continue

            content = str(b.content) if b.content is not None else ""
            page = int(b.page) if (isinstance(b.page, (int, float)) and b.page >= 1) else 1
            chapter = str(b.chapter).strip() if b.chapter else "Unknown"
            section = str(b.section).strip() if b.section else "General"
            heading_path = str(b.heading_path).strip() if b.heading_path else f"{chapter} > {section}"
            keywords = [str(k) for k in b.keywords] if isinstance(b.keywords, list) else []

            c_type = b.content_type if isinstance(b.content_type, ContentType) else ContentType.TEXT

            sanitized.append(ParsedBlock(
                content_type=c_type,
                content=content,
                page=page,
                chapter=chapter,
                section=section,
                heading_path=heading_path,
                keywords=keywords,
                image_path=b.image_path,
                table_markdown=b.table_markdown
            ))
        return sanitized

    def build_chunks_and_parents(
        self,
        blocks: List[ParsedBlock]
    ) -> Tuple[List[Chunk], List[ParentSection]]:
        """
        Group blocks by parent section (chapter > section),
        apply table summarization, image captioning, sentence-preserving text chunking,
        and construct chunks with heading prefixes and metadata.
        """
        sanitized_blocks = self._sanitize_blocks(blocks)
        if not sanitized_blocks:
            logger.warning("No valid blocks to chunk for '%s'", self.book_title)
            return [], []

        # 1. Group blocks by parent section key
        section_groups: Dict[Tuple[str, str], List[ParsedBlock]] = {}
        for block in sanitized_blocks:
            key = (block.chapter, block.section)
            if key not in section_groups:
                section_groups[key] = []
            section_groups[key].append(block)

        all_chunks: List[Chunk] = []
        all_parents: List[ParentSection] = []

        chunk_counter = 0

        for (chapter, section), sec_blocks in section_groups.items():
            heading_path = sec_blocks[0].heading_path or f"{chapter} > {section}"
            parent_id_raw = f"{self.book_title}_{chapter}_{section}"
            parent_id = generate_id("parent", parent_id_raw)

            valid_pages = [b.page for b in sec_blocks if isinstance(b.page, int) and b.page >= 1]
            page_start = min(valid_pages) if valid_pages else 1
            page_end = max(valid_pages) if valid_pages else page_start

            section_chunk_ids: List[str] = []
            section_text_pieces: List[str] = []

            # Process blocks in this section
            # Consecutive text/list blocks can be batched for coherent text splitting
            current_text_batch: List[str] = []
            current_batch_pages: List[int] = []
            current_batch_keywords: List[str] = []

            def flush_text_batch():
                nonlocal chunk_counter, all_chunks, section_chunk_ids
                if not current_text_batch:
                    return
                combined_text = "\n\n".join(current_text_batch).strip()
                if not combined_text:
                    current_text_batch.clear()
                    current_batch_pages.clear()
                    current_batch_keywords.clear()
                    return

                # Split text inside section
                sub_chunks = self.text_splitter.split_text(combined_text)
                batch_page = current_batch_pages[0] if current_batch_pages else page_start

                for sub in sub_chunks:
                    clean_sub = sub.strip()
                    if not clean_sub:
                        continue
                    chunk_counter += 1
                    cid = f"chunk_{chunk_counter:04d}_{generate_id('c', clean_sub)}"

                    # Prefix with heading path
                    prefixed_content = f"[{heading_path}]\n\n{clean_sub}"
                    
                    # Merge keywords
                    chunk_keywords = extract_keywords_from_text(
                        clean_sub,
                        additional_terms=current_batch_keywords
                    )

                    meta = ChunkMetadata(
                        book_title=self.book_title,
                        chapter=chapter,
                        section=section,
                        page=batch_page,
                        type=ContentType.TEXT.value,
                        chunk_id=cid,
                        parent_id=parent_id,
                        keywords=chunk_keywords,
                        heading_path=heading_path
                    )
                    all_chunks.append(Chunk(chunk_id=cid, content=prefixed_content, metadata=meta))
                    section_chunk_ids.append(cid)

                current_text_batch.clear()
                current_batch_pages.clear()
                current_batch_keywords.clear()

            for block in sec_blocks:
                if block.content_type == ContentType.TABLE:
                    flush_text_batch()
                    # 3. Tables: convert to markdown, keep as ONE chunk, add 1-sentence LLM summary
                    summary = self.llm_helper.summarize_table(block.content)
                    table_content = f"[{heading_path}]\n\n**Table Summary**: {summary}\n\n{block.content}"
                    
                    chunk_counter += 1
                    cid = f"chunk_{chunk_counter:04d}_{generate_id('tab', block.content)}"

                    meta = ChunkMetadata(
                        book_title=self.book_title,
                        chapter=chapter,
                        section=section,
                        page=block.page,
                        type=ContentType.TABLE.value,
                        chunk_id=cid,
                        parent_id=parent_id,
                        keywords=block.keywords,
                        heading_path=heading_path
                    )
                    all_chunks.append(Chunk(chunk_id=cid, content=table_content, metadata=meta))
                    section_chunk_ids.append(cid)
                    section_text_pieces.append(table_content)

                elif block.content_type == ContentType.IMAGE:
                    flush_text_batch()
                    # 4. Images: generate caption with vision LLM, index caption as a chunk
                    caption = self.llm_helper.caption_image(block.image_path, page_num=block.page)
                    img_content = f"[{heading_path}]\n\n**Figure Caption**: {caption}"

                    chunk_counter += 1
                    cid = f"chunk_{chunk_counter:04d}_{generate_id('img', block.image_path or str(chunk_counter))}"

                    meta = ChunkMetadata(
                        book_title=self.book_title,
                        chapter=chapter,
                        section=section,
                        page=block.page,
                        type=ContentType.IMAGE.value,
                        chunk_id=cid,
                        parent_id=parent_id,
                        keywords=block.keywords + ["Figure", "Image"],
                        heading_path=heading_path,
                        image_path=block.image_path
                    )
                    all_chunks.append(Chunk(chunk_id=cid, content=img_content, metadata=meta))
                    section_chunk_ids.append(cid)
                    section_text_pieces.append(img_content)

                elif block.content_type == ContentType.CODE:
                    flush_text_batch()
                    chunk_counter += 1
                    cid = f"chunk_{chunk_counter:04d}_{generate_id('code', block.content)}"
                    code_content = f"[{heading_path}]\n\n{block.content}"

                    meta = ChunkMetadata(
                        book_title=self.book_title,
                        chapter=chapter,
                        section=section,
                        page=block.page,
                        type=ContentType.CODE.value,
                        chunk_id=cid,
                        parent_id=parent_id,
                        keywords=block.keywords,
                        heading_path=heading_path
                    )
                    all_chunks.append(Chunk(chunk_id=cid, content=code_content, metadata=meta))
                    section_chunk_ids.append(cid)
                    section_text_pieces.append(code_content)

                else:
                    # Regular text or list
                    if block.content and block.content.strip():
                        current_text_batch.append(block.content)
                        current_batch_pages.append(block.page)
                        current_batch_keywords.extend(block.keywords)
                        section_text_pieces.append(block.content)

            # Flush any remaining text in this section
            flush_text_batch()

            # Create ParentSection
            parent_sec = ParentSection(
                parent_id=parent_id,
                book_title=self.book_title,
                chapter=chapter,
                section=section,
                page_start=page_start,
                page_end=page_end,
                heading_path=heading_path,
                full_content="\n\n".join(section_text_pieces),
                chunk_ids=section_chunk_ids
            )
            all_parents.append(parent_sec)

        logger.info(
            "Constructed %d chunks and %d parent sections for '%s'",
            len(all_chunks),
            len(all_parents),
            self.book_title
        )
        return all_chunks, all_parents
