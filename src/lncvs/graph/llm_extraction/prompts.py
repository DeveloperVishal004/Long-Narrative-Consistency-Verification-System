"""Versioned extraction prompts (frozen G2 interface §5/§6/§9).

PROMPT_VERSION is derived from the templates' own content, mirroring
lncvs.reasoning.decomposition.prompts.PROMPT_VERSION exactly -- it updates
automatically whenever either template changes, no hand-maintained version
string to forget to bump. It is recorded as audit provenance (e.g. in a
later stage's build manifest), not folded into the structured-completion
cache key directly -- CachingStructuredLLMClient's cache key is
(config.fingerprint(), schema_version, rendered_prompt), and the rendered
prompt already contains this template's text, so a template edit changes
the prompt string and therefore the cache key automatically.
"""

import hashlib

# Hackathon cost optimization: event extraction is disabled here (and in
# json_schema.py's EXTRACTION_JSON_SCHEMA) because, per the architectural
# verification on record, EventRecord/EventParticipation never reach
# EntityGraph.from_records()/GraphIndex/GraphRetriever -- events were the
# single largest source of output tokens in this pipeline for zero
# retrieval benefit. Only entity/relation instructions remain below.
SYSTEM_PROMPT = """You are a precise information-extraction system for narrative text. You extract entities and relationships EXPLICITLY stated in the provided text passage, and nothing else.

Critical rules:
1. Every entity and relation you output MUST include at least one evidence_quotes entry that is copied VERBATIM, character-for-character, from the passage below. Do not paraphrase, summarize, or correct the quote in any way -- copy it exactly as it appears, including capitalization and punctuation.
2. If you cannot find an exact quote supporting a fact, DO NOT output that fact. Omission is always preferable to fabrication.
3. Do not use any knowledge about these characters, places, or events from outside the provided passage. If the passage does not state something, it does not exist for this task, even if you recognize the work it is from.
4. Only output relation_type values from the provided closed list. If a relationship does not fit any listed type, omit it -- do not invent a new type or force a near-fit.
5. local_id values ("e1", "e2", ...) must be unique within your response and are local to this passage only -- they do not need to match any other passage.
6. Resolve pronouns and clear referring expressions to the entity they refer to within this passage, but do not guess across sentences where the referent is ambiguous -- omit the relation rather than guess.
7. Output must conform exactly to the provided JSON schema. Do not include any text, explanation, or markdown outside the JSON object."""

_USER_PROMPT_TEMPLATE = """Extract entities and relations from the following passage. This is chapter {chapter_index}{window_suffix} of a novel.

PASSAGE:
\"\"\"
{window_text}
\"\"\"

Remember: every fact requires a verbatim evidence_quotes entry from the passage above. If in doubt, omit the fact."""

PROMPT_VERSION = hashlib.sha256((SYSTEM_PROMPT + _USER_PROMPT_TEMPLATE).encode("utf-8")).hexdigest()[:8]


def render_user_prompt(window_text: str, chapter_index: int, window_index: int | None) -> str:
    """Render the user prompt for a single extraction window.

    window_index=None (an unsplit chapter/fallback segment) renders with
    no window suffix; a sub-window of a long chapter renders ", window N"
    so the model has the same locality context a human reader would.
    """
    window_suffix = "" if window_index is None else f", window {window_index}"
    return _USER_PROMPT_TEMPLATE.format(
        chapter_index=chapter_index, window_suffix=window_suffix, window_text=window_text
    )
