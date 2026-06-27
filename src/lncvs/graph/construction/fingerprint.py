"""Graph fingerprint: a canonical content hash over sorted nodes and edges
(including provenance), per the frozen G2 spec §5.

Pure function. Given the same entities/relations/events/participations,
always produces the same fingerprint, regardless of input ordering --
every collection is sorted by its own deterministic identity fields
before hashing.
"""

import hashlib

from lncvs.schemas import EntityRecord, EntityRelation, EventParticipation, EventRecord


def compute_graph_fingerprint(
    entities: tuple[EntityRecord, ...],
    relations: tuple[EntityRelation, ...],
    events: tuple[EventRecord, ...],
    participations: tuple[EventParticipation, ...],
) -> str:
    lines: list[str] = []

    for entity in sorted(entities, key=lambda e: e.entity_id):
        provenance = ",".join(f"{p.chunk_id}:{p.char_start}:{p.char_end}" for p in entity.provenance)
        lines.append(f"E:{entity.entity_id}:{entity.canonical_name}:{entity.entity_type.value}:{provenance}")

    for relation in sorted(relations, key=lambda r: (r.subject_entity_id, r.object_entity_id, r.relation_type.value)):
        lines.append(f"R:{relation.subject_entity_id}:{relation.object_entity_id}:{relation.relation_type.value}:{relation.weight}")

    for event in sorted(events, key=lambda v: v.event_id):
        lines.append(f"V:{event.event_id}:{event.predicate}:{event.temporal_kind.value}:{event.temporal_expression or ''}")

    for participation in sorted(participations, key=lambda p: (p.entity_id, p.event_id, p.role.value)):
        lines.append(f"P:{participation.entity_id}:{participation.event_id}:{participation.role.value}:{participation.weight}")

    digest_input = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()
