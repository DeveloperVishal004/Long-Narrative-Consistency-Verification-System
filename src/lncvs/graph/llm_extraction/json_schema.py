"""The literal extraction JSON schema (frozen G2 interface §1) plus its
content-hash version.

EXTRACTION_JSON_SCHEMA is the schema *body* only -- the
{"type": "object", ...} value -- not the {"name", "strict", "schema": ...}
wrapper a provider's response_format expects. OpenAIStructuredClient
builds that wrapper itself from whatever response_schema it's given
(see lncvs.llm.openai_structured_client), so this constant stays
provider-agnostic and is exactly what gets re-used as the Pydantic
validation target on the response side.

SCHEMA_VERSION is derived from the schema's own content (sorted-key JSON),
so it updates automatically whenever the schema changes -- no hand-
maintained version string to forget to bump, the same discipline
PROMPT_VERSION already follows in reasoning/decomposition/prompts.py.

Hackathon cost optimization (events removed from the wire schema): the
"events" property was removed from this dict so Gemini is never asked to
produce events at all, eliminating the single largest source of output
tokens in this pipeline -- per the architectural verification on record,
EventRecord/EventParticipation never reach EntityGraph.from_records(),
GraphIndex, GraphRetriever, or anything downstream of graph construction,
so this has zero effect on retrieval, fusion, NLI, or the final verdict.
RawEvent/RawParticipant/RawTemporal and WindowExtraction.events (schema.py)
are deliberately left defined -- WindowExtraction.events defaults to ()
and a response with no "events" key validates the same as one with an
empty list, so graph construction needs no special-casing. Re-enabling
event extraction later is reverting this dict, not a new feature.
"""

import hashlib
import json

EXTRACTION_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entities", "relations"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["local_id", "name", "type", "aliases", "evidence_quotes"],
                "properties": {
                    "local_id": {"type": "string", "pattern": "^e[0-9]+$"},
                    "name": {"type": "string", "minLength": 1},
                    "type": {"type": "string", "enum": ["PERSON", "LOCATION", "ORGANIZATION", "OBJECT", "OTHER"]},
                    "aliases": {"type": "array", "items": {"type": "string", "minLength": 1}},
                    "evidence_quotes": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                },
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["subject_local_id", "object_local_id", "relation_type", "evidence_quotes"],
                "properties": {
                    "subject_local_id": {"type": "string", "pattern": "^e[0-9]+$"},
                    "object_local_id": {"type": "string", "pattern": "^e[0-9]+$"},
                    "relation_type": {
                        "type": "string",
                        "enum": ["FAMILY_OF", "ALLY_OF", "ENEMY_OF", "MEMBER_OF", "LOCATED_AT", "POSSESSES", "SAME_AS"],
                    },
                    "evidence_quotes": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                },
            },
        },
    },
}

SCHEMA_VERSION = hashlib.sha256(json.dumps(EXTRACTION_JSON_SCHEMA, sort_keys=True).encode("utf-8")).hexdigest()[:8]
