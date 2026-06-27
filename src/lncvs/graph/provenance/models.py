"""Provenance-resolution result types (Phase 8 / G2 Slice 4).

ResolvedFact/RejectedFact wrap the original typed raw extraction fact
(never a dict -- the union of the three Raw* types from
lncvs.graph.llm_extraction.schema) alongside its resolution outcome.
These are intermediate types consumed by entity resolution (Slice 5) and
graph construction (Slice 6); nothing outside lncvs.graph holds one.

A ResolvedFact's provenance is guaranteed non-empty by construction (see
service.py) -- a fact with zero resolved chunk_ids is never wrapped in a
ResolvedFact, only ever in a RejectedFact. This is what lets a later
stage's "no node/edge without provenance" invariant be a type-level
guarantee rather than a runtime check repeated at every call site.
"""

from pydantic import BaseModel, ConfigDict, Field

from lncvs.graph.llm_extraction.schema import RawEntityMention, RawEvent, RawRelation
from lncvs.graph.provenance.matching import QuoteMatch
from lncvs.schemas import Provenance

RawFact = RawEntityMention | RawRelation | RawEvent


class ResolvedFact(BaseModel):
    """A raw fact whose evidence_quotes resolved to at least one chunk.

    chapter_index/window_index identify which window this fact's raw.local_id
    (and, for relations/events, its referenced local_ids) are scoped to --
    local_id values are window-unique, not globally unique, so entity
    resolution (Slice 5) needs this to disambiguate "e1" in one window from
    "e1" in another when building the local-id-to-global-entity-id mapping.
    Added in Slice 5 alongside RejectedFact, which already carried both.
    """

    model_config = ConfigDict(frozen=True)

    raw: RawFact
    chapter_index: int = Field(..., ge=0)
    window_index: int | None = Field(default=None, ge=0)
    provenance: tuple[Provenance, ...] = Field(..., min_length=1)
    quote_matches: tuple[QuoteMatch, ...] = Field(..., min_length=1)


class RejectedFact(BaseModel):
    """A raw fact whose evidence_quotes resolved to zero chunks --
    quarantined, never passed to entity resolution or the graph builder."""

    model_config = ConfigDict(frozen=True)

    raw: RawFact
    chapter_index: int = Field(..., ge=0)
    window_index: int | None = Field(default=None, ge=0)
    quote_matches: tuple[QuoteMatch, ...] = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)


class WindowProvenanceResult(BaseModel):
    """The complete provenance-resolution outcome for one extraction window."""

    model_config = ConfigDict(frozen=True)

    chapter_index: int = Field(..., ge=0)
    window_index: int | None = Field(default=None, ge=0)
    resolved_entities: tuple[ResolvedFact, ...] = Field(default=())
    resolved_relations: tuple[ResolvedFact, ...] = Field(default=())
    resolved_events: tuple[ResolvedFact, ...] = Field(default=())
    rejected_entities: tuple[RejectedFact, ...] = Field(default=())
    rejected_relations: tuple[RejectedFact, ...] = Field(default=())
    rejected_events: tuple[RejectedFact, ...] = Field(default=())
