"""Entity resolution result type (Phase 8 / G2 Slice 5).

A plain, frozen dataclass rather than a Pydantic model: this is an
internal handoff structure between entity resolution and graph
construction (both within lncvs.graph), never serialized, never crossing
into schemas/-governed domain-contract territory (EntityRecord itself,
which this *contains*, is the actual typed contract).
"""

from dataclasses import dataclass

from lncvs.schemas import EntityRecord

LocalEntityKey = tuple[int, int | None, str]  # (chapter_index, window_index, local_id)


@dataclass(frozen=True)
class EntityResolutionResult:
    """The merged global entities plus the mapping needed to translate a
    relation/event's window-local local_id references (Slice 6) into the
    global entity_id they were resolved to.

    local_to_global intentionally omits any local_id whose entity mention
    was quarantined in Slice 4 (never resolved) -- callers must treat a
    missing key as "this reference cannot be honored" and handle it
    explicitly (e.g. quarantine the referencing relation/event too), never
    assume the key exists.
    """

    entities: tuple[EntityRecord, ...]
    local_to_global: dict[LocalEntityKey, str]

    def resolve_local_id(self, chapter_index: int, window_index: int | None, local_id: str) -> str | None:
        """Return the global entity_id for this window-local reference, or
        None if it was never resolved (quarantined or unknown)."""
        return self.local_to_global.get((chapter_index, window_index, local_id))
