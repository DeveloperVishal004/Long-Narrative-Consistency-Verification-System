"""resolve_window_provenance: the full integration of tiered quote
matching + chunk-overlap resolution + quarantine, exercising every
required scenario end-to-end.

Boundary-dependent expectations (which chunks a given quote's span
overlaps) are computed via the already independently-tested
chunks_overlapping_span oracle (see tests/chunking/test_span_overlap.py)
rather than hand-derived by inspection -- offset arithmetic across
overlapping chunks is too error-prone to trust by eye, and this still
validates resolve_window_provenance's own logic (quote resolution, span
clipping, resolved/rejected partitioning), which is strictly more than
what the oracle itself computes.
"""

import pytest

from lncvs.chunking import chunks_overlapping_span
from lncvs.graph.llm_extraction.schema import RawEntityMention, RawEvent, RawParticipant, RawRelation, WindowExtraction
from lncvs.graph.provenance.matching import MatchTier
from lncvs.graph.provenance.models import RejectedFact
from lncvs.graph.provenance.service import resolve_window_provenance
from lncvs.schemas import DocumentChunk, EntityType, ParticipantRole, RelationType

WINDOW_TEXT = (
    "John lost his left arm in an accident in 2010 near the old bridge by the river. "
    "He never fully recovered from the shock of that day, though he tried to forget. "
    "John moved to London in 2012 and started a new and quieter life there alone."
)
WINDOW_CHAR_START = 1000  # this window begins at offset 1000 in the full document


def _chunk(chunk_id: str, start: int, end: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=WINDOW_TEXT[start - WINDOW_CHAR_START : end - WINDOW_CHAR_START],
        char_start=start,
        char_end=end,
        source_id="doc",
    )


# chunk-a/chunk-b overlap in [1040,1050); chunk-b/chunk-c overlap in [1170,1170) is
# empty by construction below -- chunk-b ends exactly where chunk-c begins, so a
# quote must be deliberately chosen to straddle [1170) to exercise that boundary.
CHUNKS = [
    _chunk("chunk-a", 1000, 1050),
    _chunk("chunk-b", 1040, 1170),
    _chunk("chunk-c", 1170, 1236),
]


def _expected_chunks(local_start: int, local_end: int) -> set[str]:
    return chunks_overlapping_span(WINDOW_CHAR_START + local_start, WINDOW_CHAR_START + local_end, CHUNKS)


def _entity(local_id: str, name: str, quotes: tuple[str, ...]) -> RawEntityMention:
    return RawEntityMention(local_id=local_id, name=name, type=EntityType.PERSON, evidence_quotes=quotes)


def _local_span(quote: str) -> tuple[int, int]:
    idx = WINDOW_TEXT.index(quote)
    return idx, idx + len(quote)


def test_resolved_entity_with_quote_fully_inside_one_chunk() -> None:
    quote = "quieter life there alone"  # tail of the window, well past any overlap
    extraction = WindowExtraction(entities=(_entity("e1", "John", (quote,)),))
    result = resolve_window_provenance(extraction, WINDOW_TEXT, 1, None, WINDOW_CHAR_START, CHUNKS)

    assert len(result.resolved_entities) == 1
    resolved = result.resolved_entities[0]
    expected = _expected_chunks(*_local_span(quote))
    assert expected == {"chunk-c"}  # sanity-check the fixture itself
    assert {p.chunk_id for p in resolved.provenance} == expected


def test_quote_in_overlap_region_resolves_to_both_overlapping_chunks() -> None:
    quote = "to forget. John moved to London"  # deliberately straddles the b/c boundary at 1170
    extraction = WindowExtraction(entities=(_entity("e1", "John", (quote,)),))
    result = resolve_window_provenance(extraction, WINDOW_TEXT, 1, None, WINDOW_CHAR_START, CHUNKS)

    resolved = result.resolved_entities[0]
    expected = _expected_chunks(*_local_span(quote))
    assert len(expected) >= 2  # sanity-check the fixture straddles a real boundary
    assert {p.chunk_id for p in resolved.provenance} == expected


def test_quote_spanning_all_three_chunks() -> None:
    extraction = WindowExtraction(entities=(_entity("e1", "John", (WINDOW_TEXT,)),))
    result = resolve_window_provenance(extraction, WINDOW_TEXT, 1, None, WINDOW_CHAR_START, CHUNKS)

    resolved = result.resolved_entities[0]
    assert {p.chunk_id for p in resolved.provenance} == {"chunk-a", "chunk-b", "chunk-c"}


def test_multiple_quotes_produce_provenance_from_multiple_chunks() -> None:
    quote_1 = "John lost his left arm in an accident in 2010"
    quote_2 = "quieter life there alone"
    extraction = WindowExtraction(entities=(_entity("e1", "John", (quote_1, quote_2)),))
    result = resolve_window_provenance(extraction, WINDOW_TEXT, 1, None, WINDOW_CHAR_START, CHUNKS)

    resolved = result.resolved_entities[0]
    expected = _expected_chunks(*_local_span(quote_1)) | _expected_chunks(*_local_span(quote_2))
    assert "chunk-a" in expected and "chunk-c" in expected  # sanity-check the fixture
    assert {p.chunk_id for p in resolved.provenance} == expected
    assert len(resolved.quote_matches) == 2
    assert all(m.tier is MatchTier.EXACT for m in resolved.quote_matches)


def test_fact_with_unresolvable_quote_is_quarantined() -> None:
    extraction = WindowExtraction(entities=(_entity("e1", "Zorblax", ("Zorblax traveled to the moon in a balloon",)),))
    result = resolve_window_provenance(extraction, WINDOW_TEXT, 3, 1, WINDOW_CHAR_START, CHUNKS)

    assert len(result.resolved_entities) == 0
    assert len(result.rejected_entities) == 1
    rejected = result.rejected_entities[0]
    assert isinstance(rejected, RejectedFact)
    assert rejected.chapter_index == 3
    assert rejected.window_index == 1
    assert rejected.quote_matches[0].tier is MatchTier.FAILED
    assert "no evidence_quotes resolved" in rejected.reason


def test_relations_and_events_are_resolved_and_quarantined_independently() -> None:
    extraction = WindowExtraction(
        entities=(_entity("e1", "John", ("John moved to London",)), _entity("e2", "London", ("to London in 2012",))),
        relations=(
            RawRelation(
                subject_local_id="e1",
                object_local_id="e2",
                relation_type=RelationType.LOCATED_AT,
                evidence_quotes=("John moved to London in 2012",),
            ),
            RawRelation(
                subject_local_id="e1",
                object_local_id="e2",
                relation_type=RelationType.ALLY_OF,
                evidence_quotes=("a sentence that never appears anywhere",),
            ),
        ),
        events=(
            RawEvent(
                local_id="v1",
                predicate="lose",
                participants=(RawParticipant(entity_local_id="e1", role=ParticipantRole.PATIENT),),
                evidence_quotes=("John lost his left arm in an accident in 2010",),
            ),
        ),
    )
    result = resolve_window_provenance(extraction, WINDOW_TEXT, 1, None, WINDOW_CHAR_START, CHUNKS)

    assert len(result.resolved_entities) == 2
    assert len(result.resolved_relations) == 1
    assert len(result.rejected_relations) == 1
    assert len(result.resolved_events) == 1
    assert len(result.rejected_events) == 0


def test_resolution_is_deterministic_across_independent_calls() -> None:
    extraction = WindowExtraction(
        entities=(_entity("e1", "John", ("John moved to London in 2012", "John lost his left arm")),)
    )
    first = resolve_window_provenance(extraction, WINDOW_TEXT, 1, None, WINDOW_CHAR_START, CHUNKS)
    second = resolve_window_provenance(extraction, WINDOW_TEXT, 1, None, WINDOW_CHAR_START, CHUNKS)
    assert first == second


def test_rejects_empty_chunk_list() -> None:
    extraction = WindowExtraction(entities=(_entity("e1", "John", ("John moved",)),))
    with pytest.raises(ValueError):
        resolve_window_provenance(extraction, WINDOW_TEXT, 1, None, WINDOW_CHAR_START, [])
