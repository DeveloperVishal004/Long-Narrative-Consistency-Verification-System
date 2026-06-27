"""Versioned question generation prompt template.

PROMPT_VERSION is derived from the template's own content (see
lncvs.reasoning.decomposition.prompts for the same pattern and the
rationale for why this is provenance metadata, not a cache key).
"""

import hashlib

PROMPT_TEMPLATE = """You are generating retrieval-oriented probe questions for a single atomic claim, \
as part of a narrative consistency verification system.

Rules:
- Output ONLY a JSON array of strings, nothing else.
- Each string is one yes/no or short-answer question about the claim's subject.
- Questions must genuinely ask, never assert a new fact as already true. \
("Did John lose an arm?" is correct; "John lost his arm in 2010." is not — it asserts an unstated fact.)
- Questions should probe for information that, if true, would CONTRADICT the claim — \
this is the entire purpose: surfacing evidence that is not semantically similar to the \
claim itself but would still be relevant to verifying it.
- Stay strictly on the claim's subject and topic. Do not invent unrelated entities, places, \
or events that have nothing to do with the claim.
- If you cannot think of any useful probe question, output an empty JSON array: []

Atomic claim: "{claim}"

JSON array of probe questions:"""

PROMPT_VERSION = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:8]


def render_question_generation_prompt(atomic_claim_text: str) -> str:
    """Render the question generation prompt for a single atomic claim."""
    return PROMPT_TEMPLATE.format(claim=atomic_claim_text)
