"""The literal fact-verification JSON schema (Phase H3) plus its
content-hash version, mirroring lncvs.graph.llm_extraction.json_schema's
discipline exactly.

FACT_VERIFICATION_JSON_SCHEMA is the schema *body* only, provider-agnostic,
sent via StructuredLLMClient.complete_structured(). "quotes" deliberately
has no minItems (defaults to 0): a NOT_MENTIONED verdict legitimately has
zero quotes, and the strict-mode schema must permit that shape -- the
verifier's own logic (llm_verifier.py), not the schema, enforces that
SUPPORTED/CONTRADICTED requires at least one quote.
"""

import hashlib
import json

FACT_VERIFICATION_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "confidence", "quotes", "explanation"],
    "properties": {
        "verdict": {"type": "string", "enum": ["SUPPORTED", "CONTRADICTED", "NOT_MENTIONED"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "quotes": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "explanation": {"type": "string", "minLength": 1},
    },
}

SCHEMA_VERSION = hashlib.sha256(json.dumps(FACT_VERIFICATION_JSON_SCHEMA, sort_keys=True).encode("utf-8")).hexdigest()[:8]
