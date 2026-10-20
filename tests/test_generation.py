"""Unit and integration tests for Phase 4 Generation."""

import pytest
from generation.prompt_templates import SYSTEM_PROMPT, format_context_from_hits, build_user_prompt
from generation.generator import AnswerGenerator
from generation.qa_chain import BookRAGPipeline


def test_system_prompt_adherence():
    """Verify system prompt contains all required instructions from the user specification."""
    assert "friendly expert tutor" in SYSTEM_PROMPT
    assert "Answer ONLY using the provided context" in SYSTEM_PROMPT
    assert "conversational language" in SYSTEM_PROMPT
    assert "If the context has a table, present it clearly" in SYSTEM_PROMPT
    assert "mention it by its caption and page" in SYSTEM_PROMPT
    assert "End with sources in the form (Chapter, Page)" in SYSTEM_PROMPT
    assert "I couldn't find that in this book" in SYSTEM_PROMPT
    assert "Never invent facts" in SYSTEM_PROMPT


def test_context_formatting():
    """Verify hits are formatted into clear context blocks with metadata."""
    hits = [
        {
            "chunk_id": "c1",
            "content": "Superposition state description.",
            "metadata": {"chapter": "Chapter 1", "section": "Section 1", "page": 1, "type": "text"}
        }
    ]
    formatted = format_context_from_hits(hits)
    assert "Chapter 1" in formatted
    assert "Page: 1" in formatted
    assert "Superposition state description." in formatted


def test_not_found_handling():
    """Verify generator returns standard not found answer when no hits exist."""
    gen = AnswerGenerator()
    res = gen.generate(question="What is string theory?", hits=[])
    assert res.answer == "I couldn't find that in this book."
    assert len(res.sources) == 0


def test_generator_synthesis_and_citations():
    """Verify generator extracts sources and produces response with (Chapter, Page) citation."""
    gen = AnswerGenerator()
    hits = [
        {
            "chunk_id": "c1",
            "content": "[Chapter 1 > Section 1.1]\n\nA qubit is defined as the fundamental unit of quantum information.",
            "metadata": {"chapter": "Chapter 1", "section": "Section 1.1", "page": 1, "type": "text"}
        }
    ]
    res = gen.generate(question="What is a qubit?", hits=hits)
    assert len(res.sources) == 1
    assert res.sources[0].chapter == "Chapter 1"
    assert res.sources[0].page == 1
    assert "qubit" in res.answer.lower()
    assert "(Chapter 1, Page 1)" in res.answer


def test_end_to_end_pipeline_query():
    """Verify full BookRAGPipeline runs on indexed book."""
    pipeline = BookRAGPipeline()
    response = pipeline.query("What is a qubit and superposition?")
    assert response.answer != ""
    assert len(response.retrieved_chunks) > 0
    assert len(response.sources) > 0
