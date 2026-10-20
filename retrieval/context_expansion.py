"""Context expansion module to enrich short retrieved chunks with their parent section."""

import re
import logging
from typing import List, Dict, Any, Optional

from ingest.parent_store import ParentStore

logger = logging.getLogger(__name__)

DEFAULT_SHORT_THRESHOLD = 250  # Characters


class ContextExpander:
    """Expands retrieved chunks to their full parent section when content is brief."""

    def __init__(self, parent_store: Optional[ParentStore] = None, threshold_chars: int = DEFAULT_SHORT_THRESHOLD):
        self.parent_store = parent_store or ParentStore()
        self.threshold_chars = threshold_chars

    def _strip_heading_prefix(self, content: str) -> str:
        """Strip leading [Heading Path] prefix to measure raw content length."""
        return re.sub(r"^\[.*?\]\s*", "", content, flags=re.DOTALL).strip()

    def expand_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check each hit; if raw content length < threshold_chars,
        retrieve its parent section from ParentStore and expand its content.
        """
        expanded_hits: List[Dict[str, Any]] = []

        for hit in hits:
            hit_copy = dict(hit)
            content = hit_copy.get("content", "")
            raw_text = self._strip_heading_prefix(content)
            metadata = hit_copy.get("metadata", {})
            parent_id = metadata.get("parent_id")

            # Check if chunk is too short and parent is available
            if len(raw_text) < self.threshold_chars and parent_id:
                parent = self.parent_store.get(parent_id)
                if parent and parent.full_content and len(parent.full_content) > len(content):
                    logger.info("Expanding short chunk '%s' (%d chars) to parent section '%s' (%d chars)",
                                hit_copy.get("chunk_id"), len(raw_text), parent_id, len(parent.full_content))
                    hit_copy["original_content"] = content
                    # Retain heading prefix with full parent section
                    heading_prefix = f"[{parent.heading_path}]\n\n" if parent.heading_path else ""
                    hit_copy["content"] = f"{heading_prefix}{parent.full_content}"
                    metadata["is_expanded"] = True
                    metadata["parent_heading"] = parent.heading_path
                else:
                    metadata["is_expanded"] = False
            else:
                metadata["is_expanded"] = False

            hit_copy["metadata"] = metadata
            expanded_hits.append(hit_copy)

        return expanded_hits
