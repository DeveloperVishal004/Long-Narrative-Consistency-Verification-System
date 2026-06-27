"""LLMFactVerifier: the LLM-backed FactVerifier implementation (Phase H3,
redesigned to evidence-SET level per the explicit architectural review on
record).

Redesign summary (was: evidence-level, one LLM call per (claim, evidence
chunk) pair, mirroring CrossEncoderFactVerifier exactly -- see this
class's prior revision for that design): LLMFactVerifier now makes
exactly ONE LLM call per (claim, complete evidence set) -- the model
reasons across all retrieved evidence for the fact jointly and returns
exactly one FactVerification. CrossEncoderFactVerifier is unaffected and
continues to operate at evidence-level exactly as before; the FactVerifier
protocol is unaffected (it returns list[FactVerification] either way --
a length-N list at evidence-level, a length-1 list at evidence-set-level
-- and to_nli_results/classify/ThresholdRuleEngine handle both
identically, since neither was ever written assuming a particular list
length per claim).

Rationale (architectural review on record): evidence-level verification
cannot connect support/contradiction that spans more than one retrieved
chunk, and lets one noisy or borderline chunk's isolated judgment flip an
entire claim under classify()'s "any contradiction dominates" rule before
the model ever sees the other 9 chunks for context. Reading the complete
evidence set in one call lets the model weigh all retrieved passages
together before committing to a single verdict. This is not the LLM
producing a final verdict: it is still scoped to one atomic fact's
SUPPORTED/CONTRADICTED/NOT_MENTIONED relationship to its own evidence --
exactly the same role NLI verification already plays today -- and
ThresholdRuleEngine still independently applies the confidence threshold
and the across-CLAIMS Rule 1>2>3 precedence, completely unchanged.

This is still the trust-boundary component CLAUDE.md's hallucination-
prevention discipline exists for: every quote the model claims supports
or contradicts a fact is independently re-verified as an exact substring
of SOME evidence record in the retrieved set (not necessarily the first),
using the same tiered matcher already built and frozen for graph
extraction (lncvs.graph.provenance.matching.resolve_quote), restricted to
Tier 1 (EXACT) only.

Disclosed consequence of the redesign: FactVerification.evidence_chunk_id
now means "an anchor chunk for this verdict" rather than "the one chunk
this verdict is about" -- it is the chunk containing the first verified
quote (SUPPORTED/CONTRADICTED), or the first evidence record by rank
(NOT_MENTIONED, which has no quote to anchor to). This is a real,
disclosed narrowing of that field's meaning, not a bug.

Failures (malformed JSON, a SUPPORTED/CONTRADICTED verdict with no quote,
or any quote that fails verification against every evidence record in the
set) raise ValueError immediately and are never silently downgraded to
NOT_MENTIONED here -- unchanged from the original Phase H3 requirement.
"""

from lncvs.graph.provenance.matching import MatchTier, resolve_quote
from lncvs.llm import StructuredLLMClient
from lncvs.reasoning.fact_verification.llm_config import FactVerificationConfig
from lncvs.reasoning.fact_verification.llm_prompts import render_fact_verification_prompt
from lncvs.reasoning.fact_verification.llm_raw import RawFactVerdict, parse_fact_verdict
from lncvs.reasoning.fact_verification.llm_schema import FACT_VERIFICATION_JSON_SCHEMA
from lncvs.schemas import AtomicClaim, FactVerification, FactVerificationLabel, FusedEvidence

_LABELS_REQUIRING_A_QUOTE = frozenset({FactVerificationLabel.SUPPORTED, FactVerificationLabel.CONTRADICTED})


class LLMFactVerifier:
    """FactVerifier backed by an injected StructuredLLMClient.

    Holds no state beyond its injected dependencies. Determinism rests
    entirely on the StructuredLLMClient (wrap in CachingStructuredLLMClient
    for real provider calls, exactly as graph extraction does; a fake is
    deterministic by construction for tests).
    """

    def __init__(self, client: StructuredLLMClient, config: FactVerificationConfig | None = None) -> None:
        self._client = client
        self._config = config or FactVerificationConfig()

    def verify(self, claim: AtomicClaim, evidence: list[FusedEvidence]) -> list[FactVerification]:
        """Return exactly one FactVerification reasoning across the
        COMPLETE evidence set for this claim (evidence-SET-level, the
        Phase H redesign -- CrossEncoderFactVerifier remains evidence-level,
        unaffected by this change).

        An empty evidence list returns an empty list, unchanged from the
        original evidence-level contract -- still the correct input for a
        claim with no retrieved evidence, never an error.
        """
        if not evidence:
            return []
        return [self._verify_evidence_set(claim, evidence)]

    def _verify_evidence_set(self, claim: AtomicClaim, evidence: list[FusedEvidence]) -> FactVerification:
        prompt = render_fact_verification_prompt(claim.text, [record.text for record in evidence])
        completion = self._client.complete_structured(prompt, FACT_VERIFICATION_JSON_SCHEMA)
        raw = parse_fact_verdict(completion.data)
        return self._to_fact_verification(raw, claim, evidence)

    def _to_fact_verification(self, raw: RawFactVerdict, claim: AtomicClaim, evidence: list[FusedEvidence]) -> FactVerification:
        label = FactVerificationLabel(raw.verdict)

        if label in _LABELS_REQUIRING_A_QUOTE and not raw.quotes:
            raise ValueError(
                f"{label.value} verdict for claim {claim.claim_id!r} (evidence set of {len(evidence)} chunks) "
                "supplied zero quotes; SUPPORTED/CONTRADICTED must cite at least one verbatim quote."
            )

        verified_quotes, anchor_chunk_id = self._verify_quotes_against_evidence_set(raw.quotes, evidence, claim)

        return FactVerification(
            atomic_claim_id=claim.claim_id,
            evidence_chunk_id=anchor_chunk_id,
            label=label,
            confidence=raw.confidence,
            supporting_quotes=verified_quotes,
            explanation=raw.explanation,
        )

    def _verify_quotes_against_evidence_set(
        self, quotes: tuple[str, ...], evidence: list[FusedEvidence], claim: AtomicClaim
    ) -> tuple[tuple[str, ...], str]:
        """Re-verify every claimed quote as an exact (Tier 1 only) substring
        of SOME evidence record in the set -- not necessarily the first.
        Raises on the FIRST quote that matches none of them; a
        hallucinated citation is rejected outright, never silently
        dropped or downgraded.

        Returns (quotes, anchor_chunk_id): anchor_chunk_id is the chunk_id
        of the record containing the first verified quote, or -- when
        quotes is empty (NOT_MENTIONED) -- the first evidence record by
        rank, a deterministic default since no single chunk is "the" one
        a NOT_MENTIONED verdict can point to.
        """
        anchor_chunk_id = evidence[0].chunk_id
        for index, quote in enumerate(quotes):
            matched_chunk_id = self._find_chunk_containing_quote(quote, evidence)
            if matched_chunk_id is None:
                raise ValueError(
                    f"Quote {quote!r} for claim {claim.claim_id!r} could not be verified as an exact "
                    "substring of ANY evidence record in the retrieved set -- rejecting the response "
                    "rather than accepting a hallucinated citation."
                )
            if index == 0:
                anchor_chunk_id = matched_chunk_id
        return quotes, anchor_chunk_id

    def _find_chunk_containing_quote(self, quote: str, evidence: list[FusedEvidence]) -> str | None:
        for record in evidence:
            if resolve_quote(quote, record.text).tier is MatchTier.EXACT:
                return record.chunk_id
        return None
