"""Versioned claim decomposition prompt template.

PROMPT_VERSION is derived from the template's own content, so it updates
automatically whenever the template text changes — no hand-maintained
version string to forget to bump. It is recorded in DecompositionConfig as
provenance metadata; it is not part of the cache key (see lncvs.llm.cache
docstring for why the rendered prompt text already makes that unnecessary).
"""

import hashlib

PROMPT_TEMPLATE = """You are decomposing a narrative claim into atomic, self-contained factual assertions for consistency verification.

Rules:
- Output ONLY a JSON array of strings, nothing else.
- Each string is one atomic claim: a single, indivisible factual assertion.
- Resolve all pronouns and implicit references so each atomic claim is understandable on its own (e.g. use "John" instead of "he").
- Do not add any information that is not stated or directly implied by the original claim.
- Do not omit any factual assertion present in the original claim.

Original claim: "{claim}"

JSON array of atomic claims:"""

PROMPT_VERSION = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:8]


def render_decomposition_prompt(claim: str) -> str:
    """Render the decomposition prompt for a single original claim."""
    return PROMPT_TEMPLATE.format(claim=claim)
