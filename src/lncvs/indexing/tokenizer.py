"""Shared tokenizer for BM25 corpus and query text.

A single tokenize() function used on both sides of BM25Index (indexing and
querying) makes corpus/query tokenization divergence structurally
impossible — a class of silent recall failure flagged in earlier reviews.
Pure, no external dependencies, no model loading.
"""

import re

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-token split. Deterministic and dependency-free."""
    return _TOKEN_PATTERN.findall(text.lower())
