"""Deterministic content-hash ID derivation for the graph subsystem.

Pure functions, no graph/model calls involved -- mirrors
lncvs.retrieval.identity and lncvs.chunking's chunk_id derivation: the same
input always produces the same ID, never uuid4() or another random source.
"""

import hashlib


def make_entity_id(canonical_name: str, entity_type: str) -> str:
    """Deterministic ID for an EntityRecord.

    entity_type is accepted as a plain string (the EntityType enum's
    .value) so this module stays free of a schemas import, the same
    discipline lncvs.retrieval.identity follows for RetrievalSource.
    """
    digest_input = f"{canonical_name}:{entity_type}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()[:16]


def make_raw_evidence_id(query_text: str, chunk_id: str, rank: int) -> str:
    """Claim-agnostic raw evidence_id, re-derived by RetrievalOrchestrator
    once claim/query/source provenance is known. Identical shape to
    ChromaIndex._make_evidence_id / BM25Index._make_raw_evidence_id."""
    digest_input = f"{query_text}:{chunk_id}:{rank}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()[:16]


def make_event_id(predicate: str, participant_entity_ids: list[str], anchor_chunk_id: str, anchor_char_start: int) -> str:
    """Deterministic ID for an EventRecord: hash(predicate, sorted
    participant entity_ids, anchoring chunk span) per the frozen G2 spec
    §5. participant_entity_ids is sorted by the caller's responsibility is
    not assumed -- sorted here, so callers may pass them in any order."""
    sorted_ids = ":".join(sorted(participant_entity_ids))
    digest_input = f"{predicate}:{sorted_ids}:{anchor_chunk_id}:{anchor_char_start}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()[:16]
