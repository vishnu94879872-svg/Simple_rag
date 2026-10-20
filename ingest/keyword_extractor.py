"""Keyword and Key-Term Extractor for PDF content."""

import re
from typing import List, Set

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "can", "will", "just", "should",
    "now", "figure", "table", "chapter", "section", "page", "example", "note"
}

# Regex for Markdown bold text
MARKDOWN_BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")

# Regex for definitions: "Term: definition" or "Term — definition" or 'Term' is defined as...
DEFINITION_PATTERN = re.compile(
    r"^(?:[\s\*\-\d\.]*)([A-Z][A-Za-z0-9\s\-]{2,35})\s*(?::|—|–|-)\s+[A-Za-z]",
    re.MULTILINE
)
DEFINED_AS_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9\s\-]{2,35})\s+(?:is defined as|refers to|denotes|represents)\b",
    re.IGNORECASE
)


def clean_keyword(term: str) -> str:
    """Strip formatting, extra punctuation, and whitespace from a keyword."""
    term = re.sub(r"[\*#_`\":\(\)\[\]]", "", term).strip()
    term = re.sub(r"\s+", " ", term)
    return term


def is_valid_keyword(term: str) -> bool:
    """Validate if an extracted term is meaningful."""
    cleaned = clean_keyword(term)
    if not cleaned or len(cleaned) < 3 or len(cleaned) > 50:
        return False
    if cleaned.lower() in STOP_WORDS:
        return False
    # Avoid numeric-only strings
    if re.fullmatch(r"[\d\.\-\s]+", cleaned):
        return False
    return True


def extract_keywords_from_text(text: str, additional_terms: List[str] = None) -> List[str]:
    """
    Extract key terms from text content:
    - Bold markdown tokens (**term**)
    - Explicit definitions (Term: description, Term is defined as...)
    - Additional terms (e.g. from PDF font bold spans)
    """
    keywords: Set[str] = set()

    # 1. Additional terms passed in (e.g., from PDF bold spans)
    if additional_terms:
        for term in additional_terms:
            if is_valid_keyword(term):
                keywords.add(clean_keyword(term))

    # 2. Markdown bold syntax
    for match in MARKDOWN_BOLD_PATTERN.finditer(text):
        term = match.group(1) or match.group(2)
        if term and is_valid_keyword(term):
            keywords.add(clean_keyword(term))

    # 3. Heading/definition style: "Term: ..."
    for match in DEFINITION_PATTERN.finditer(text):
        term = match.group(1)
        if term and is_valid_keyword(term):
            keywords.add(clean_keyword(term))

    # 4. "X is defined as / refers to"
    for match in DEFINED_AS_PATTERN.finditer(text):
        term = match.group(1)
        if term and is_valid_keyword(term):
            keywords.add(clean_keyword(term))

    # Sort keywords by appearance or alphabetical
    return sorted(list(keywords))[:15]
