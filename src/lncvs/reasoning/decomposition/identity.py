"""Deterministic content-hash ID derivation for claim decomposition.

Pure functions, no LLM involved — identical text always produces identical
IDs, across runs and across process instances.
"""

import hashlib


def make_source_claim_id(original_claim: str) -> str:
    """Deterministic ID for an original (pre-decomposition) claim."""
    digest_input = f"source:{original_claim}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()[:16]


def make_atomic_claim_id(parent_claim_id: str, index: int, text: str) -> str:
    """Deterministic ID for a single atomic claim within a decomposition."""
    digest_input = f"{parent_claim_id}:{index}:{text}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()[:16]
