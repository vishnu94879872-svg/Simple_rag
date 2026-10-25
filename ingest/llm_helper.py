"""LLM helper for table summarization and image captioning."""

import base64
import logging
from pathlib import Path
from typing import Optional
from config import settings

logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path: Path) -> str:
    """Read an image file and encode it as a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class LLMHelper:
    """Helper client for LLM and Vision tasks with automatic fallback."""

    def __init__(self):
        self.provider = settings.llm_provider.lower()
        self.api_key = settings.llm_api_key
        self.base_url = settings.llm_base_url
        self.model = settings.llm_model
        self.vision_model = settings.vision_model
        self._client = None

        if self.provider == "ollama" or (self.api_key and self.provider != "mock"):
            try:
                from openai import OpenAI
                # Ollama's OpenAI-compatible endpoint is local and does not need
                # an OpenAI credential; the SDK requires a non-empty placeholder.
                client_key = self.api_key or ("ollama" if self.provider == "ollama" else "")
                self._client = OpenAI(api_key=client_key, base_url=self.base_url)
            except Exception as e:
                logger.warning("Could not initialize OpenAI client: %s. Using heuristic fallback.", e)

    def summarize_table(self, table_markdown: str) -> str:
        """
        Generate a 1-sentence summary for a markdown table.
        Uses LLM if configured; otherwise provides an intelligent rule-based summary.
        """
        if not table_markdown or not table_markdown.strip():
            return "Empty data table."

        # Attempt LLM completion if client is configured
        if self._client:
            try:
                prompt = (
                    "Provide a concise, 1-sentence summary describing what information "
                    "this table conveys. Do not include markdown formatting or quotes.\n\n"
                    f"Table:\n{table_markdown[:2500]}"
                )
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise data summarizer."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.2,
                )
                summary = response.choices[0].message.content.strip()
                if summary:
                    return summary
            except Exception as e:
                logger.warning("LLM table summarization failed: %s. Falling back to heuristic.", e)

        # Heuristic fallback summary
        lines = [line.strip() for line in table_markdown.strip().splitlines() if line.strip()]
        header_line = lines[0] if lines else ""
        cols = [c.strip() for c in header_line.split("|") if c.strip()]
        num_rows = max(0, len(lines) - 2)
        if cols:
            col_preview = ", ".join(cols[:4])
            if len(cols) > 4:
                col_preview += f", and {len(cols) - 4} other columns"
            return f"Table summarizing data across {col_preview} with approximately {num_rows} entries."
        return "Data table summarizing tabular information from the text."

    def caption_image(self, image_path: Path, page_num: int = 1) -> str:
        """
        Generate a descriptive caption for an image using a Vision LLM.
        Falls back to metadata-driven description if vision LLM is unavailable.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            return f"Figure from page {page_num}."

        if self._client:
            try:
                base64_img = encode_image_to_base64(image_path)
                ext = image_path.suffix.lower().replace(".", "")
                mime_type = "image/png" if ext == "png" else "image/jpeg"

                response = self._client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this figure or image from a book in 1 to 2 informative sentences, focusing on diagrams, charts, labels, or visual concepts shown."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_img}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=150,
                    temperature=0.2
                )
                caption = response.choices[0].message.content.strip()
                if caption:
                    return caption
            except Exception as e:
                logger.warning("Vision LLM captioning failed: %s. Falling back to heuristic.", e)

        # Fallback caption
        stem = image_path.stem.replace("_", " ").title()
        return f"Illustration or diagram from page {page_num} ({stem})."
