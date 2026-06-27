"""Window-local raw extraction DTOs (Phase 8 / G2, frozen interface §2).

These are intermediate types, distinct from the final domain types in
schemas/graph.py (EntityRecord, EntityRelation, ...). They exist only
between extraction and provenance assignment/entity resolution (Slice 4+),
which consume and discard them -- nothing outside lncvs.graph.llm_extraction
ever holds a WindowExtraction. local_id values (e1, e2, ... / v1, v2, ...)
are window-unique, never global identifiers; global identity assignment
happens deterministically in a later stage, never here.

Field shapes mirror lncvs.graph.llm_extraction.json_schema.EXTRACTION_JSON_SCHEMA
exactly -- these models are both the Pydantic validation target for a raw
provider response and (via .model_json_schema()-independent hand-written
json_schema.py) the literal schema sent to the provider's structured-output
API. The two are kept as two independent, hand-written artifacts rather
than generated from one another, because the provider-side schema has
constraints (e.g. "strict": True requiring every property listed in
required) that do not map cleanly onto Pydantic's Optional/default
semantics -- see RawEvent.temporal's docstring.
"""

from pydantic import BaseModel, ConfigDict, Field

from lncvs.schemas import EntityType, ParticipantRole, RelationType, TemporalKind

_RELATION_TYPES_FROM_LLM = frozenset(
    {
        RelationType.FAMILY_OF,
        RelationType.ALLY_OF,
        RelationType.ENEMY_OF,
        RelationType.MEMBER_OF,
        RelationType.LOCATED_AT,
        RelationType.POSSESSES,
        RelationType.SAME_AS,
    }
)


class RawEntityMention(BaseModel):
    """One entity as extracted from a single window. type uses the
    EntityType vocabulary (schemas/graph.py), which includes OBJECT to
    match the extraction JSON schema's entity-type enum exactly -- no
    coercion or information loss between the two."""

    model_config = ConfigDict(frozen=True)

    local_id: str = Field(..., pattern=r"^e[0-9]+$")
    name: str = Field(..., min_length=1)
    type: EntityType = Field(..., description="Coarse entity category, as judged by the LLM for this window.")
    aliases: tuple[str, ...] = Field(default=())
    evidence_quotes: tuple[str, ...] = Field(..., min_length=1)


class RawRelation(BaseModel):
    """One entity-to-entity relation as extracted from a single window."""

    model_config = ConfigDict(frozen=True)

    # Not pattern-constrained to ^e[0-9]+$: live runs showed the LLM
    # occasionally emits a non-conforming placeholder (e.g. "OTHER") when
    # it can't pin a relation's subject/object down to a real local entity.
    # Rejecting the whole window's extraction over one bad reference is
    # worse than letting it through -- lncvs.graph.construction already
    # resolves local IDs against the window's known entities and quarantines
    # anything that doesn't match as a dangling reference, which is the
    # correct place for this check, not Pydantic validation at parse time.
    subject_local_id: str = Field(..., min_length=1)
    object_local_id: str = Field(..., min_length=1)
    relation_type: RelationType = Field(..., description="Must be one of the G2 LLM-extracted relation types.")
    evidence_quotes: tuple[str, ...] = Field(..., min_length=1)

    def model_post_init(self, __context: object) -> None:
        if self.relation_type not in _RELATION_TYPES_FROM_LLM:
            raise ValueError(
                f"relation_type {self.relation_type!r} is not a valid LLM-extracted relation type "
                f"(CO_OCCURS is reserved for the G1 deterministic builder, never the LLM extractor)"
            )


class RawParticipant(BaseModel):
    """One entity's role in one event, as extracted from a single window."""

    model_config = ConfigDict(frozen=True)

    # Same non-conforming-placeholder issue as RawRelation's local IDs --
    # quarantined downstream as a dangling reference, not rejected here.
    entity_local_id: str = Field(..., min_length=1)
    role: ParticipantRole


class RawTemporal(BaseModel):
    """An event's temporal attribute. The provider's strict structured-output
    mode requires every object property to be listed in "required" (no
    truly optional keys), so a temporal object with no useful expression is
    represented as kind=NONE with time_expression=None, rather than by
    omitting the temporal key -- the JSON schema makes the whole temporal
    field nullable (object | null) for the same reason, and RawEvent.temporal
    is typed Optional to match."""

    model_config = ConfigDict(frozen=True)

    time_expression: str | None = Field(default=None)
    kind: TemporalKind


class RawEvent(BaseModel):
    """One event as extracted from a single window."""

    model_config = ConfigDict(frozen=True)

    local_id: str = Field(..., pattern=r"^v[0-9]+$")
    predicate: str = Field(..., min_length=1)
    participants: tuple[RawParticipant, ...] = Field(..., min_length=1)
    temporal: RawTemporal | None = Field(default=None)
    evidence_quotes: tuple[str, ...] = Field(..., min_length=1)


class WindowExtraction(BaseModel):
    """The complete validated extraction result for a single window."""

    model_config = ConfigDict(frozen=True)

    entities: tuple[RawEntityMention, ...] = Field(default=())
    relations: tuple[RawRelation, ...] = Field(default=())
    events: tuple[RawEvent, ...] = Field(default=())
