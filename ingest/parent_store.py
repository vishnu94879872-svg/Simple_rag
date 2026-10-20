"""Parent Section Store for hierarchical context expansion."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, List

from ingest.models import ParentSection
from config import settings

logger = logging.getLogger(__name__)


class ParentStore:
    """Manages persistence and retrieval of parent section texts."""

    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = store_dir or settings.parent_store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._sections: Dict[str, ParentSection] = {}
        self._load_all()

    def _file_path(self, parent_id: str) -> Path:
        # Sanitize filename
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in parent_id)
        return self.store_dir / f"{safe_id}.json"

    def _load_all(self) -> None:
        """Load all parent sections found in storage directory."""
        count = 0
        for json_file in self.store_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sec = ParentSection(**data)
                    self._sections[sec.parent_id] = sec
                    count += 1
            except Exception as e:
                logger.warning("Failed to load parent section from %s: %s", json_file, e)
        if count > 0:
            logger.info("Loaded %d parent sections from %s", count, self.store_dir)

    def save(self, section: ParentSection) -> None:
        """Save a single parent section to memory and disk."""
        self._sections[section.parent_id] = section
        out_file = self._file_path(section.parent_id)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(section.model_dump_json(indent=2))

    def save_all(self, sections: List[ParentSection]) -> None:
        """Save multiple parent sections."""
        for sec in sections:
            self.save(sec)

    def get(self, parent_id: str) -> Optional[ParentSection]:
        """Retrieve a parent section by its parent_id."""
        return self._sections.get(parent_id)

    def list_all(self) -> List[ParentSection]:
        """Return all stored parent sections."""
        return list(self._sections.values())
