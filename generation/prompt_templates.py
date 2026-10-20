"""Prompt templates and context builders for BookRAG generation."""

from typing import List, Dict, Any

SYSTEM_PROMPT = (
    "You are a friendly expert tutor who has read this book. Answer ONLY using "
    "the provided context. Explain in simple, natural, conversational language, "
    "not copied text. If the context has a table, present it clearly. If an image "
    "is relevant, mention it by its caption and page. End with sources in the form "
    "(Chapter, Page). If the answer is not in the context, say: 'I couldn't find "
    "that in this book.' Never invent facts."
)


def format_context_from_hits(hits: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a clear, structured context string for the LLM."""
    if not hits:
        return "No relevant context found in the book."

    context_blocks = []
    for idx, hit in enumerate(hits, 1):
        meta = hit.get("metadata", {})
        chapter = meta.get("chapter", "Unknown Chapter")
        section = meta.get("section", "Unknown Section")
        page = meta.get("page", "?")
        c_type = meta.get("type", "text")
        content = hit.get("content", "").strip()

        block_header = f"--- Context Chunk {idx} [Chapter: {chapter} | Section: {section} | Page: {page} | Type: {c_type}] ---"
        context_blocks.append(f"{block_header}\n{content}")

    return "\n\n".join(context_blocks)


def build_user_prompt(question: str, context_text: str) -> str:
    """Construct the user prompt combining book context and user question."""
    return (
        f"Context from the book:\n"
        f"========================\n"
        f"{context_text}\n"
        f"========================\n\n"
        f"Question: {question.strip()}\n\n"
        f"Helpful Tutor Answer:"
    )
