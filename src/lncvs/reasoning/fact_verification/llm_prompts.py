"""Versioned fact-verification prompts (Phase H3, redesigned to
evidence-SET level per the explicit architectural review and approval --
see llm_verifier.py's module docstring for the full rationale).

PROMPT_VERSION is derived from both templates' own content, mirroring
lncvs.graph.llm_extraction.prompts.PROMPT_VERSION exactly -- it updates
automatically whenever either template changes; the rendered prompt
(already containing this template's text) is what CachingStructuredLLMClient
hashes, so a template edit self-invalidates the cache with no separate
version-bumping discipline required.

The user prompt embeds ONLY the atomic fact text and the retrieved
evidence passages -- never the original backstory the fact was decomposed
from. This is deliberate, not an oversight: the verifier must judge the
fact purely against the retrieved evidence, the same way NLIVerifier's
premise is fixed as the evidence text alone (lncvs.reasoning.nli.service).

Redesign note: this prompt now presents ALL retrieved evidence passages
for one atomic fact in a single call (previously: one passage per call).
The model is explicitly instructed to reason across the complete set
before deciding, rather than judging any single passage in isolation --
the entire point of the redesign (see the architectural review on
record: this lets the model connect evidence split across passages, and
weigh a single ambiguous passage against the other passages in the set
before committing to a verdict, instead of one isolated judgment per
chunk being enough to flip a claim).
"""

import hashlib

SYSTEM_PROMPT = """You are a precise fact-verification system for narrative consistency checking. You are given ONE atomic fact and a SET of evidence passages retrieved from a novel, all retrieved as candidates for verifying this same fact. Read ALL passages together and decide the fact's relationship to the COMPLETE set, not to any single passage in isolation.

Critical rules:
1. NOT_MENTIONED is the default, safe choice. If NONE of the passages address the fact, you MUST return NOT_MENTIONED -- never CONTRADICTED. Absence of evidence is NEVER contradiction.
2. Only return CONTRADICTED if at least one passage explicitly states something that is incompatible with the fact. A passage that is merely silent on the fact, or about a different topic, is NOT a contradiction. If some passages are irrelevant and one is genuinely silent while none contradict, the verdict is still NOT_MENTIONED unless an explicit contradiction exists somewhere in the set.
3. Only return SUPPORTED if the passages, taken together or individually, explicitly state or directly entail the fact. Evidence may be split across more than one passage -- read all of them before deciding NOT_MENTIONED.
4. If you return SUPPORTED or CONTRADICTED, you MUST include at least one quote in "quotes" copied VERBATIM, character-for-character, from one of the passages below -- do not paraphrase, summarize, or correct it in any way. Do not invent or alter the quote in any way. A quote may be drawn from any one of the passages, not necessarily the first.
5. If you return NOT_MENTIONED, "quotes" MUST be an empty array. Never fabricate a quote to justify a NOT_MENTIONED verdict, and never fabricate a quote to justify any other verdict either.
6. Use ONLY the passages given below. Do not use any outside knowledge of the novel, its characters, or its plot, even if you recognize the work.
7. Output must conform exactly to the provided JSON schema. Do not include any text, explanation, or markdown outside the JSON object."""

_USER_PROMPT_TEMPLATE = """ATOMIC FACT:
"{fact_text}"

You are given {n_passages} evidence passage(s) retrieved from the novel as candidates for verifying this fact. Read ALL of them before deciding.

{passages_block}

Classify the atomic fact's relationship to the COMPLETE evidence set above as exactly one of SUPPORTED, CONTRADICTED, or NOT_MENTIONED. Remember: if no passage addresses the fact, the answer is NOT_MENTIONED, not CONTRADICTED. Base your verdict on the full set together, not any single passage in isolation."""

_PASSAGE_BLOCK_TEMPLATE = """PASSAGE {index}:
\"\"\"
{passage_text}
\"\"\""""

PROMPT_VERSION = hashlib.sha256((SYSTEM_PROMPT + _USER_PROMPT_TEMPLATE + _PASSAGE_BLOCK_TEMPLATE).encode("utf-8")).hexdigest()[:8]


def render_fact_verification_prompt(fact_text: str, evidence_texts: list[str]) -> str:
    """Render the user prompt for verifying one atomic fact against the
    COMPLETE set of retrieved evidence passages in a single call. Never
    receives the original backstory the fact was decomposed from -- only
    the already-isolated atomic fact text and the evidence passages.

    evidence_texts must be non-empty -- the caller (LLMFactVerifier.verify)
    never calls this for a claim with zero retrieved evidence; that case
    short-circuits to an empty result list before any prompt is rendered.
    """
    if not evidence_texts:
        raise ValueError("evidence_texts must not be empty")

    passages_block = "\n\n".join(
        _PASSAGE_BLOCK_TEMPLATE.format(index=index, passage_text=text) for index, text in enumerate(evidence_texts, start=1)
    )
    return _USER_PROMPT_TEMPLATE.format(fact_text=fact_text, n_passages=len(evidence_texts), passages_block=passages_block)
