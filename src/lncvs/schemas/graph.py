"""Graph subsystem content types (Phase 8 / Version 2).

Per the Phase 8 architecture review: the graph never leaves the chunk-ID
space, and Chunk is metadata (via Provenance), never a node. Two node
content types exist: EntityRecord (Stage G1 onward) and EventRecord
(added in G2 Slice 6). Two edge content types: EntityRelation (entity to
entity, carrying a RelationType) and EventParticipation (entity to event,
carrying a ParticipantRole -- the PARTICIPATES_IN edge). All four are
immutable and append-only: once constructed during graph build, they are
never mutated.

These types are defined here, not in lncvs.graph, because CLAUDE.md's
Required Core Models list designates schemas/ as the only place shared
data contracts may live -- lncvs.graph.models re-exports these names
without defining competing types, exactly as every other module's
models.py does.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from lncvs.schemas.provenance import Provenance


class EntityType(str, Enum):
    """Closed vocabulary for EntityRecord.entity_type.

    Stage G1's extractor is deterministic and rule-based (capitalized-token
    extraction, no NER model) and therefore cannot responsibly distinguish
    a person from a place -- assigning PERSON/LOCATION without a real
    classifier would be fabrication, not extraction. Every Stage G1 entity
    is OTHER. PERSON/LOCATION/ORGANIZATION/OBJECT become reachable once a
    real extractor is introduced (Phase 8 / G2's LLM extraction pipeline).

    OBJECT was added during G2 Slice 3 implementation, not in the original
    G1 design: the frozen G2 extraction JSON schema's entity-type enum
    includes OBJECT (e.g. a ship's name, a will, a weapon -- real,
    distinguishing categories in both project novels), and this enum must
    match it exactly or every OBJECT-typed extraction would fail Pydantic
    validation for a reason that has nothing to do with extraction quality.
    Flagged and approved as an additive extension, analogous to the
    RelationType extension for the same G2 vocabulary.
    """

    PERSON = "PERSON"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    OBJECT = "OBJECT"
    OTHER = "OTHER"


class RelationType(str, Enum):
    """Closed vocabulary for EntityRelation.relation_type.

    CO_OCCURS is Stage G1's value: a deterministic, NLP-free proxy signal
    meaning "these two entities were mentioned in the same chunk" --
    explicitly not a claim that the source text asserts a typed
    relationship between them. It is kept for G1 backward compatibility;
    the G1 deterministic builder is the only producer of CO_OCCURS edges.

    The remaining values are the G2 architecture freeze's typed relation
    vocabulary, extracted by the LLM extraction pipeline (Phase 8 / G2) and
    never by the G1 builder. Per the G2 architecture-freeze decision (G2
    Decision 2), this is a single closed vocabulary for "edge type between
    two entities" rather than a second, separate enum -- CO_OCCURS and the
    typed G2 values are never both emitted by the same builder for the same
    edge. SAME_AS means the source text explicitly asserts two named
    entities are one and the same (e.g. an alias reveal); per the frozen
    G2 entity-resolution policy it is a joining edge, not a merge trigger.
    """

    CO_OCCURS = "CO_OCCURS"
    FAMILY_OF = "FAMILY_OF"
    ALLY_OF = "ALLY_OF"
    ENEMY_OF = "ENEMY_OF"
    MEMBER_OF = "MEMBER_OF"
    LOCATED_AT = "LOCATED_AT"
    POSSESSES = "POSSESSES"
    SAME_AS = "SAME_AS"


class ParticipantRole(str, Enum):
    """Closed vocabulary for an entity's role in an event (Phase 8 / G2).

    Used both by the raw LLM extraction DTOs (lncvs.graph.llm_extraction)
    and, once an event graph is built (a later G2 stage), by the
    PARTICIPATES_IN edge's role attribute -- one shared vocabulary, not a
    separate raw/final pair, since the role concept does not change
    meaning between extraction and the final graph.
    """

    AGENT = "AGENT"
    PATIENT = "PATIENT"
    EXPERIENCER = "EXPERIENCER"
    LOCATION = "LOCATION"
    INSTRUMENT = "INSTRUMENT"


class TemporalKind(str, Enum):
    """Closed vocabulary for an event's temporal attribute (Phase 8 / G2).

    Per the frozen G2 spec, temporal information is captured as an event
    attribute and stored, but does not yet drive traversable temporal
    edges (PRECEDES remains deferred) -- this enum exists to type that
    stored attribute, nothing more.
    """

    ABSOLUTE = "ABSOLUTE"
    RELATIVE = "RELATIVE"
    NONE = "NONE"


class EntityRecord(BaseModel):
    """A resolved narrative entity -- the graph's only Stage G1 node type.

    entity_id is a deterministic content hash (see
    lncvs.graph.identity.make_entity_id), never uuid4(), so rebuilding the
    graph from identical chunks always reproduces identical entity_ids.
    provenance lists every chunk this entity was mentioned in -- this list
    is the entity's only link back into chunk space; there is no separate
    ChunkNode to traverse to.
    """

    model_config = ConfigDict(frozen=True)

    entity_id: str = Field(..., min_length=1, description="Deterministic content-hash identifier.")
    canonical_name: str = Field(..., min_length=1, description="Normalized, deduplicated entity name.")
    entity_type: EntityType = Field(..., description="Coarse entity category.")
    provenance: tuple[Provenance, ...] = Field(
        ..., min_length=1, description="Every chunk/span this entity was mentioned in."
    )


class EventRecord(BaseModel):
    """A validated event with typed participants (Phase 8 / G2 Slice 6).

    event_id is a deterministic content hash over (predicate, sorted
    participant entity_ids, anchoring chunk span) -- see
    lncvs.graph.identity.make_event_id. temporal_kind/temporal_expression
    are stored as attributes only; they do not yet drive a traversable
    PRECEDES edge (deferred, per the frozen spec). provenance is the
    event's own resolved span(s), independent of EventParticipation's
    (which share the same provenance, since RawEvent's evidence_quotes are
    not broken down per-participant).
    """

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(..., min_length=1, description="Deterministic content-hash identifier.")
    predicate: str = Field(..., min_length=1, description="Normalized lemma of the event's head verb.")
    temporal_kind: TemporalKind = Field(..., description="Whether/how this event is temporally anchored.")
    temporal_expression: str | None = Field(default=None, description="Verbatim time expression, if any.")
    provenance: tuple[Provenance, ...] = Field(..., min_length=1, description="Every chunk this event was extracted from.")


class EventParticipation(BaseModel):
    """A PARTICIPATES_IN edge: one entity's role in one event.

    weight is the count of distinct provenance chunks supporting this
    specific (entity, event, role) participation -- the same
    corroboration-strength role EntityRelation.weight plays.
    """

    model_config = ConfigDict(frozen=True)

    entity_id: str = Field(..., min_length=1)
    event_id: str = Field(..., min_length=1)
    role: ParticipantRole = Field(..., description="This entity's role in the event.")
    weight: int = Field(..., gt=0, description="Count of distinct provenance chunks supporting this participation.")
    provenance: tuple[Provenance, ...] = Field(..., min_length=1)


class EntityRelation(BaseModel):
    """A typed, weighted, provenance-bearing edge between two entities.

    weight is the count of distinct provenance chunks supporting this
    relation -- corroboration strength, used by graph retrieval's chunk
    scoring formula.

    Direction convention depends on relation_type, and this is load-bearing:
      - CO_OCCURS (G1's symmetric, NLP-free proxy signal): subject_entity_id/
        object_entity_id are stored in deterministic ascending-entity-ID
        order regardless of extraction order, so the same co-occurring
        pair always produces the same edge identity no matter which entity
        was encountered first. Direction carries no meaning for this type.
      - Every other (G2, LLM-extracted) value: relations are directional
        per the frozen G2 spec ("direction: directed, subject -> object"),
        and subject_entity_id/object_entity_id are stored EXACTLY as
        resolved from the extraction's subject_local_id/object_local_id --
        never reordered. Reordering would silently invert the meaning of
        an asymmetric relation like POSSESSES or LOCATED_AT for any pair
        whose true object happens to sort below its true subject. This was
        caught and fixed during G2 Slice 6 implementation, before it ever
        reached a constructed graph -- see lncvs.graph.construction.service.
    """

    model_config = ConfigDict(frozen=True)

    subject_entity_id: str = Field(..., min_length=1, description="Ascending-sorted first entity_id of the pair.")
    object_entity_id: str = Field(..., min_length=1, description="Ascending-sorted second entity_id of the pair.")
    relation_type: RelationType = Field(..., description="Typed relation between the two entities.")
    weight: int = Field(..., gt=0, description="Count of distinct provenance chunks supporting this relation.")
    provenance: tuple[Provenance, ...] = Field(
        ..., min_length=1, description="Every chunk where this relation was observed."
    )
