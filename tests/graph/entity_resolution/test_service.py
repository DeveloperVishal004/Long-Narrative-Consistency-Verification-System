"""resolve_entities: the full Slice 5 entry point -- merged EntityRecords
plus the local-id-to-global-entity-id mapping."""

from lncvs.graph.entity_resolution.models import EntityResolutionResult
from lncvs.graph.entity_resolution.service import resolve_entities
from lncvs.graph.llm_extraction.schema import RawEntityMention
from lncvs.graph.provenance.matching import MatchTier, QuoteMatch
from lncvs.graph.provenance.models import ResolvedFact
from lncvs.schemas import EntityRecord, EntityType, Provenance


def _mention(local_id: str, name: str, chapter_index: int, window_index: int | None, chunk_id: str, char_start: int) -> ResolvedFact:
    raw = RawEntityMention(local_id=local_id, name=name, type=EntityType.PERSON, evidence_quotes=(name,))
    provenance = (Provenance(chunk_id=chunk_id, char_start=char_start, char_end=char_start + len(name)),)
    quote_match = QuoteMatch(quote=name, tier=MatchTier.EXACT, char_start=0, char_end=len(name))
    return ResolvedFact(raw=raw, chapter_index=chapter_index, window_index=window_index, provenance=provenance, quote_matches=(quote_match,))


def test_resolve_entities_returns_one_entity_per_distinct_name() -> None:
    facts = [
        _mention("e1", "John", 0, None, "c1", 0),
        _mention("e1", "Mary", 0, None, "c1", 10),
        _mention("e2", "John", 1, None, "c2", 0),
    ]
    result = resolve_entities(facts)
    assert isinstance(result, EntityResolutionResult)
    assert len(result.entities) == 2
    assert all(isinstance(e, EntityRecord) for e in result.entities)


def test_local_ids_with_the_same_string_in_different_windows_resolve_to_correct_distinct_or_same_global_id() -> None:
    """"e1" in window (0,None) is John; "e1" in window (1,None) is Mary --
    same local_id string, different windows, must NOT collapse."""
    facts = [
        _mention("e1", "John", 0, None, "c1", 0),
        _mention("e1", "Mary", 1, None, "c2", 0),
    ]
    result = resolve_entities(facts)
    john_id = result.resolve_local_id(0, None, "e1")
    mary_id = result.resolve_local_id(1, None, "e1")
    assert john_id is not None
    assert mary_id is not None
    assert john_id != mary_id


def test_same_entity_across_windows_maps_every_local_reference_to_the_same_global_id() -> None:
    facts = [
        _mention("e3", "John", 0, None, "c1", 0),
        _mention("e7", "John", 4, 1, "c2", 0),
    ]
    result = resolve_entities(facts)
    id_a = result.resolve_local_id(0, None, "e3")
    id_b = result.resolve_local_id(4, 1, "e7")
    assert id_a == id_b
    assert len(result.entities) == 1


def test_unknown_local_reference_resolves_to_none() -> None:
    facts = [_mention("e1", "John", 0, None, "c1", 0)]
    result = resolve_entities(facts)
    assert result.resolve_local_id(0, None, "e99") is None
    assert result.resolve_local_id(99, None, "e1") is None


def test_quarantined_entity_never_appears_in_the_mapping() -> None:
    """A quarantined entity (Slice 4) is simply never passed to
    resolve_entities -- its local_id is correctly absent from the mapping."""
    facts = [_mention("e1", "John", 0, None, "c1", 0)]  # "e2" was quarantined upstream, never included
    result = resolve_entities(facts)
    assert result.resolve_local_id(0, None, "e2") is None


def test_empty_input_produces_empty_result() -> None:
    result = resolve_entities([])
    assert result.entities == ()
    assert result.local_to_global == {}


def test_resolve_entities_is_deterministic_across_calls() -> None:
    facts = [
        _mention("e1", "John", 0, None, "c1", 0),
        _mention("e2", "Mary", 1, None, "c2", 10),
        _mention("e3", "John", 2, None, "c3", 20),
    ]
    first = resolve_entities(facts)
    second = resolve_entities(facts)
    assert first.entities == second.entities
    assert first.local_to_global == second.local_to_global


def test_entities_are_returned_sorted_by_entity_id() -> None:
    facts = [
        _mention("e1", "Zelda", 0, None, "c1", 0),
        _mention("e2", "Amy", 1, None, "c2", 0),
        _mention("e3", "Mark", 2, None, "c3", 0),
    ]
    result = resolve_entities(facts)
    ids = [e.entity_id for e in result.entities]
    assert ids == sorted(ids)
