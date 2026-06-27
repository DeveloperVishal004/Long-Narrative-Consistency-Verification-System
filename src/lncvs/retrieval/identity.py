"""Deterministic content-hash ID derivation for retrieval integration.

Pure functions, no retriever calls involved.
"""

import hashlib


def make_query_id(atomic_claim_id: str, origin: str, question_id: str, text: str) -> str:
    """Deterministic ID for a single RetrievalQuery.

    origin and question_id are passed as plain strings (not the QueryOrigin
    enum) to keep this module free of a schemas import beyond what's needed;
    callers pass question_id="" for CLAIM-origin queries.
    """
    digest_input = f"{atomic_claim_id}:{origin}:{question_id}:{text}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()[:16]


def make_evidence_id(query_id: str, source: str, chunk_id: str, rank: int) -> str:
    """Deterministic evidence_id incorporating query_id and source.

    Folding query_id in (rather than raw query text) is what prevents a
    collision when two different atomic claims happen to generate the
    identical query text: their query_ids still differ because query_id
    itself is derived from (atomic_claim_id, origin, question_id, text).

    Folding source in (Phase 4) is what prevents a collision when the same
    query retrieves the same chunk at the same rank from two different
    retrieval backends (e.g. semantic and lexical) — without source in the
    hash, those two distinct evidence records would receive identical IDs.
    source is passed as a plain string (the RetrievalSource enum's .value)
    to keep this module free of a schemas import beyond what's needed.
    """
    digest_input = f"{query_id}:{source}:{chunk_id}:{rank}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()[:16]
