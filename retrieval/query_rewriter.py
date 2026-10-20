"""Query rewriter using LLM to transform conversational queries into standalone search queries."""

import logging
from typing import List, Dict, Optional

from config import settings
from ingest.llm_helper import LLMHelper

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a search query optimizer. Given a conversation history and a follow-up user question, rewrite the question into a clear, specific, self-contained standalone query suitable for vector and keyword search in a book knowledge base.
Rules:
1. Do NOT answer the question.
2. Resolve any ambiguous pronouns (it, that, they, he, she) using the chat history.
3. Keep the query concise, keyword-rich, and natural.
4. Output ONLY the rewritten standalone query, nothing else."""


class QueryRewriter:
    """Rewrites user queries into standalone search queries using conversation context."""

    def __init__(self, llm_helper: Optional[LLMHelper] = None):
        self.llm_helper = llm_helper or LLMHelper()

    def rewrite(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Rewrite user query into a clearer standalone search query.
        If no chat history is provided or LLM is offline, returns clean original query.
        """
        query_stripped = query.strip()
        if not query_stripped:
            return ""

        # If no history and query is already fairly detailed, return as is
        if not chat_history and len(query_stripped.split()) >= 4:
            return query_stripped

        # If LLM client is available
        if self.llm_helper and self.llm_helper._client:
            try:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                if chat_history:
                    for turn in chat_history[-6:]:  # Use last 3 turns
                        messages.append({
                            "role": turn.get("role", "user"),
                            "content": turn.get("content", "")
                        })
                messages.append({"role": "user", "content": f"User question: {query_stripped}"})

                response = self.llm_helper._client.chat.completions.create(
                    model=settings.llm_model,
                    messages=messages,
                    max_tokens=60,
                    temperature=0.0
                )
                rewritten = response.choices[0].message.content.strip()
                # Remove quotes if enclosed
                rewritten = rewritten.strip('"\'')
                if rewritten:
                    logger.info("Rewrote query: '%s' -> '%s'", query_stripped, rewritten)
                    return rewritten
            except Exception as e:
                logger.warning("Query rewriting LLM call failed: %s. Using original query.", e)

        # Fallback heuristic: if history is present and query has pronouns
        if chat_history and any(p in query_stripped.lower().split() for p in ["it", "this", "that", "they", "its"]):
            last_user_msg = next((m["content"] for m in reversed(chat_history) if m.get("role") == "user"), "")
            if last_user_msg:
                return f"{last_user_msg} {query_stripped}".strip()

        return query_stripped
