"""Graph construction result type (Phase 8 / G2 Slice 6).

A plain, frozen dataclass, the same pattern lncvs.graph.entity_resolution.models
uses -- this is the internal handoff structure construction produces, not
a schemas/-governed domain contract (the EntityRecord/EntityRelation/
EventRecord/EventParticipation it contains are the actual contracts).

Scope decision, explicitly disclosed (not silently descoped): events are
constructed here as fully typed, correctly-provenanced, content-deduplicated
data -- but are kept as a *separate* structure alongside entities/relations,
not mixed into lncvs.graph.builder.EntityGraph's existing networkx
structure. This is what lets Slice 8 wire entities+relations into the
unchanged, already-tested G1 GraphIndex/traversal code immediately (the
working end-to-end retrieval pipeline), while event-aware traversal
remains a disclosed, deferred enhancement rather than a silent scope cut.
"""

from dataclasses import dataclass

from lncvs.graph.provenance.models import RejectedFact
from lncvs.schemas import EntityRecord, EntityRelation, EventParticipation, EventRecord


@dataclass(frozen=True)
class ConstructedGraph:
    entities: tuple[EntityRecord, ...]
    relations: tuple[EntityRelation, ...]
    events: tuple[EventRecord, ...]
    participations: tuple[EventParticipation, ...]
    rejected_relations: tuple[RejectedFact, ...]
    rejected_events: tuple[RejectedFact, ...]
    fingerprint: str
