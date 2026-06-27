"""Deterministic content-hash ID derivation for question generation.

Pure functions, no LLM involved. Deliberately independent of
lncvs.reasoning.decomposition.identity — sibling reasoning/ submodules do
not import each other's internals, per CLAUDE.md's dependency rules.
"""

import hashlib


def make_question_id(atomic_claim_id: str, index: int, text: str) -> str:
    """Deterministic ID for a single probe question generated for a claim."""
    digest_input = f"{atomic_claim_id}:{index}:{text}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()[:16]
