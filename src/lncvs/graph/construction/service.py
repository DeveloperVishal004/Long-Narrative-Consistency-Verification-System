"""Graph construction service (Phase 8 / G2 Slice 6): assembles resolved
relations and events into the final typed EntityRelation/EventRecord/
EventParticipation content, resolving every local-id reference through
Slice 5's EntityResolutionResult and quarantining anything dangling.

This is a second trust boundary, parallel to provenance assignment's: a
relation or event reaches the graph only if every entity it references
was itself successfully resolved. A reference to a quarantined (Slice 4)
or otherwise-unknown local_id is dropped here, never silently ignored or
defaulted -- collected into rejected_relations/rejected_events instead.
"""

from collections import defaultdict

from lncvs.graph.construction.fingerprint import compute_graph_fingerprint
from lncvs.graph.construction.models import ConstructedGraph
from lncvs.graph.entity_resolution.models import EntityResolutionResult
from lncvs.graph.identity import make_event_id
from lncvs.graph.llm_extraction.schema import RawEvent, RawRelation
from lncvs.graph.provenance.models import RejectedFact, ResolvedFact
from lncvs.schemas import EntityRelation, EventParticipation, EventRecord, ParticipantRole, Provenance, TemporalKind


def _fact_sort_key(fact: ResolvedFact) -> tuple[int, int, str]:
    """Deterministic tiebreak order for picking among duplicate facts that
    collapsed onto the same edge/event identity."""
    return (fact.chapter_index, fact.window_index if fact.window_index is not None else -1, fact.raw.local_id)


def _dedupe_provenance(facts: list[ResolvedFact]) -> tuple[Provenance, ...]:
    union = {provenance for fact in facts for provenance in fact.provenance}
    return tuple(sorted(union, key=lambda p: (p.chunk_id, p.char_start)))


def _build_relations(
    relation_facts: list[ResolvedFact], resolution: EntityResolutionResult
) -> tuple[tuple[EntityRelation, ...], tuple[RejectedFact, ...]]:
    groups: dict[tuple[str, str, str], list[ResolvedFact]] = defaultdict(list)
    rejected: list[RejectedFact] = []

    for fact in relation_facts:
        raw: RawRelation = fact.raw
        subject_id = resolution.resolve_local_id(fact.chapter_index, fact.window_index, raw.subject_local_id)
        object_id = resolution.resolve_local_id(fact.chapter_index, fact.window_index, raw.object_local_id)

        if subject_id is None or object_id is None:
            rejected.append(
                RejectedFact(
                    raw=raw,
                    chapter_index=fact.chapter_index,
                    window_index=fact.window_index,
                    quote_matches=fact.quote_matches,
                    reason="subject_local_id or object_local_id did not resolve to a known entity (dangling reference)",
                )
            )
            continue

        if subject_id == object_id:
            rejected.append(
                RejectedFact(
                    raw=raw,
                    chapter_index=fact.chapter_index,
                    window_index=fact.window_index,
                    quote_matches=fact.quote_matches,
                    reason="subject and object resolved to the same entity (degenerate self-relation)",
                )
            )
            continue

        # G2 relations are directional (frozen spec §3: "direction: directed,
        # subject -> object") -- store exactly as resolved from the
        # extracted local IDs, never reordered. The ascending-entity-ID
        # canonicalization that EntityRelation's docstring describes
        # applies only to symmetric G1 CO_OCCURS edges, which this builder
        # never produces.
        groups[(subject_id, object_id, raw.relation_type.value)].append(fact)

    relations = []
    for (subject_id, object_id, _relation_type_value), facts in groups.items():
        provenance = _dedupe_provenance(facts)
        weight = len({p.chunk_id for p in provenance})
        relations.append(
            EntityRelation(
                subject_entity_id=subject_id,
                object_entity_id=object_id,
                relation_type=facts[0].raw.relation_type,
                weight=weight,
                provenance=provenance,
            )
        )

    return tuple(relations), tuple(rejected)


def _build_events(
    event_facts: list[ResolvedFact], resolution: EntityResolutionResult
) -> tuple[tuple[EventRecord, ...], tuple[EventParticipation, ...], tuple[RejectedFact, ...]]:
    event_groups: dict[str, list[tuple[ResolvedFact, list[tuple[str, ParticipantRole]]]]] = defaultdict(list)
    rejected: list[RejectedFact] = []

    for fact in event_facts:
        raw: RawEvent = fact.raw
        resolved_participants: list[tuple[str, ParticipantRole]] = []
        dangling = False

        for participant in raw.participants:
            entity_id = resolution.resolve_local_id(fact.chapter_index, fact.window_index, participant.entity_local_id)
            if entity_id is None:
                dangling = True
                break
            resolved_participants.append((entity_id, participant.role))

        if dangling:
            rejected.append(
                RejectedFact(
                    raw=raw,
                    chapter_index=fact.chapter_index,
                    window_index=fact.window_index,
                    quote_matches=fact.quote_matches,
                    reason="at least one participant's entity_local_id did not resolve to a known entity (dangling reference)",
                )
            )
            continue

        anchor = fact.provenance[0]
        participant_ids = [entity_id for entity_id, _ in resolved_participants]
        event_id = make_event_id(raw.predicate, participant_ids, anchor.chunk_id, anchor.char_start)
        event_groups[event_id].append((fact, resolved_participants))

    events: list[EventRecord] = []
    participations_by_key: dict[tuple[str, str, str], list[ResolvedFact]] = defaultdict(list)

    for event_id, group in event_groups.items():
        facts = [fact for fact, _ in group]
        canonical_fact = min(facts, key=_fact_sort_key)
        raw: RawEvent = canonical_fact.raw

        provenance = _dedupe_provenance(facts)
        kind = raw.temporal.kind if raw.temporal is not None else TemporalKind.NONE
        expression = raw.temporal.time_expression if raw.temporal is not None else None

        events.append(
            EventRecord(event_id=event_id, predicate=raw.predicate, temporal_kind=kind, temporal_expression=expression, provenance=provenance)
        )

        for fact, resolved_participants in group:
            for entity_id, role in resolved_participants:
                participations_by_key[(entity_id, event_id, role.value)].append(fact)

    participations: list[EventParticipation] = []
    for (entity_id, event_id, role_value), facts in participations_by_key.items():
        provenance = _dedupe_provenance(facts)
        weight = len({p.chunk_id for p in provenance})
        participations.append(
            EventParticipation(
                entity_id=entity_id,
                event_id=event_id,
                role=ParticipantRole(role_value),
                weight=weight,
                provenance=provenance,
            )
        )

    return tuple(events), tuple(participations), tuple(rejected)


def build_graph(
    resolution: EntityResolutionResult,
    relation_facts: list[ResolvedFact],
    event_facts: list[ResolvedFact],
) -> ConstructedGraph:
    """Build the final graph content from Slice 5's resolved entities plus
    the raw resolved relation/event facts (Slice 4), quarantining any
    relation or event that references an entity local_id that never
    resolved (e.g. it was quarantined in Slice 4, or never existed)."""
    relations, rejected_relations = _build_relations(relation_facts, resolution)
    events, participations, rejected_events = _build_events(event_facts, resolution)

    fingerprint = compute_graph_fingerprint(resolution.entities, relations, events, participations)

    return ConstructedGraph(
        entities=resolution.entities,
        relations=relations,
        events=events,
        participations=participations,
        rejected_relations=rejected_relations,
        rejected_events=rejected_events,
        fingerprint=fingerprint,
    )
