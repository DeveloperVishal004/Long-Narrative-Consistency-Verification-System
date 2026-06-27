"""Entity resolution service: the public entry point for Slice 5.

resolve_entities() is the only function callers (Slice 6's graph
construction) need. It is pure -- no LLM calls, no I/O -- and fully
deterministic given identical input, which is what makes it exhaustively
unit-testable offline.
"""

from lncvs.graph.entity_resolution.merge import compute_components, merge_component
from lncvs.graph.entity_resolution.models import EntityResolutionResult
from lncvs.graph.provenance.models import ResolvedFact
from lncvs.schemas import EntityRecord


def resolve_entities(entity_facts: list[ResolvedFact]) -> EntityResolutionResult:
    """Merge entity_facts (ResolvedFact instances wrapping a RawEntityMention)
    across every window into global EntityRecords, plus the local-id-to-
    global-entity-id mapping graph construction needs to resolve
    relation/event participant references.

    entity_facts must contain only entity mentions (ResolvedFact.raw being
    a RawEntityMention) -- callers are responsible for filtering relations/
    events out before calling this function; entity resolution never looks
    at relations or events, including SAME_AS, by design (frozen spec §4).
    """
    components = compute_components(entity_facts)

    entities: list[EntityRecord] = []
    local_to_global: dict[tuple[int, int | None, str], str] = {}

    for members in components:
        entity = merge_component(members)
        entities.append(entity)
        for member in members:
            local_to_global[(member.chapter_index, member.window_index, member.raw.local_id)] = entity.entity_id

    entities.sort(key=lambda entity: entity.entity_id)

    return EntityResolutionResult(entities=tuple(entities), local_to_global=local_to_global)
