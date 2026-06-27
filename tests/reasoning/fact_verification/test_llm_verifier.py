"""LLMFactVerifier tests (evidence-SET-level redesign): protocol
conformance, exactly-one-call-per-claim behavior, the three labels, quote
verification against the COMPLETE evidence set, and -- the highest-
priority tests in this file -- the trust-boundary rejection of any
hallucinated citation, now checked against every evidence record in the
set rather than a single one."""

import pytest

from lncvs.reasoning.fact_verification import FactVerifier, LLMFactVerifier
from lncvs.schemas import AtomicClaim, FactVerificationLabel, FusedEvidence, RetrievalSource
from tests.llm.fakes import FakeStructuredLLMClient


def _claim(text: str = "John used both hands") -> AtomicClaim:
    return AtomicClaim(claim_id="claim-1", text=text)


def _fused(chunk_id: str, text: str) -> FusedEvidence:
    return FusedEvidence(
        atomic_claim_id="claim-1",
        chunk_id=chunk_id,
        text=text,
        rrf_score=0.5,
        contributing_sources=[RetrievalSource.SEMANTIC],
        contributing_query_ids=["query-1"],
    )


def test_llm_fact_verifier_satisfies_fact_verifier_protocol() -> None:
    verifier = LLMFactVerifier(FakeStructuredLLMClient(default_response={}))
    assert isinstance(verifier, FactVerifier)


def test_supported_with_valid_verbatim_quote_is_accepted() -> None:
    evidence_text = "John lost his left arm in an accident in 2010."
    fake = FakeStructuredLLMClient(
        default_response={
            "verdict": "SUPPORTED",
            "confidence": 0.95,
            "quotes": ["John lost his left arm in an accident in 2010."],
            "explanation": "The passage directly states this.",
        }
    )
    verifier = LLMFactVerifier(fake)

    results = verifier.verify(_claim("John lost his left arm"), [_fused("chunk-1", evidence_text)])

    assert len(results) == 1
    assert results[0].label is FactVerificationLabel.SUPPORTED
    assert results[0].confidence == 0.95
    assert results[0].supporting_quotes == ("John lost his left arm in an accident in 2010.",)
    assert results[0].atomic_claim_id == "claim-1"
    assert results[0].evidence_chunk_id == "chunk-1"


def test_contradicted_with_valid_verbatim_quote_is_accepted() -> None:
    evidence_text = "John never lost his arm; he played piano with both hands at the concert."
    fake = FakeStructuredLLMClient(
        default_response={
            "verdict": "CONTRADICTED",
            "confidence": 0.9,
            "quotes": ["John never lost his arm"],
            "explanation": "The passage explicitly contradicts the fact.",
        }
    )
    verifier = LLMFactVerifier(fake)

    results = verifier.verify(_claim("John lost his arm"), [_fused("chunk-1", evidence_text)])

    assert results[0].label is FactVerificationLabel.CONTRADICTED
    assert results[0].supporting_quotes == ("John never lost his arm",)


def test_not_mentioned_with_empty_quotes_is_accepted() -> None:
    fake = FakeStructuredLLMClient(
        default_response={
            "verdict": "NOT_MENTIONED",
            "confidence": 0.85,
            "quotes": [],
            "explanation": "None of the passages address this fact at all.",
        }
    )
    verifier = LLMFactVerifier(fake)

    results = verifier.verify(_claim(), [_fused("chunk-1", "John moved to Paris in 1820.")])

    assert results[0].label is FactVerificationLabel.NOT_MENTIONED
    assert results[0].supporting_quotes == ()


def test_supported_with_zero_quotes_raises() -> None:
    """SUPPORTED/CONTRADICTED must always cite at least one quote -- a
    verdict with none is malformed, not silently accepted."""
    fake = FakeStructuredLLMClient(
        default_response={"verdict": "SUPPORTED", "confidence": 0.9, "quotes": [], "explanation": "x"}
    )
    verifier = LLMFactVerifier(fake)

    with pytest.raises(ValueError, match="zero quotes"):
        verifier.verify(_claim(), [_fused("chunk-1", "some evidence text")])


def test_contradicted_with_zero_quotes_raises() -> None:
    fake = FakeStructuredLLMClient(
        default_response={"verdict": "CONTRADICTED", "confidence": 0.9, "quotes": [], "explanation": "x"}
    )
    verifier = LLMFactVerifier(fake)

    with pytest.raises(ValueError, match="zero quotes"):
        verifier.verify(_claim(), [_fused("chunk-1", "some evidence text")])


def test_hallucinated_quote_not_present_in_any_evidence_record_is_rejected() -> None:
    """The trust-boundary rejection: a quote the model invented, that does
    not appear in ANY evidence record in the set, must raise -- never be
    silently accepted as supporting evidence."""
    fake = FakeStructuredLLMClient(
        default_response={
            "verdict": "SUPPORTED",
            "confidence": 0.9,
            "quotes": ["John lost his right leg in the war."],  # not in any evidence record
            "explanation": "x",
        }
    )
    verifier = LLMFactVerifier(fake)
    evidence = [
        _fused("chunk-1", "John lost his left arm in an accident in 2010."),
        _fused("chunk-2", "John moved to London in 2012."),
    ]

    with pytest.raises(ValueError, match="could not be verified"):
        verifier.verify(_claim(), evidence)


def test_paraphrased_quote_that_is_not_an_exact_substring_is_rejected() -> None:
    """A near-miss paraphrase (not a true verbatim substring) must be
    rejected, not fuzzy-accepted -- this verifier accepts Tier 1 (exact,
    modulo canonicalization) matches only."""
    fake = FakeStructuredLLMClient(
        default_response={
            "verdict": "SUPPORTED",
            "confidence": 0.9,
            "quotes": ["John lost an arm in an accident around 2010."],  # paraphrase, not verbatim
            "explanation": "x",
        }
    )
    verifier = LLMFactVerifier(fake)

    with pytest.raises(ValueError, match="could not be verified"):
        verifier.verify(_claim(), [_fused("chunk-1", "John lost his left arm in an accident in 2010.")])


def test_quote_with_smart_quote_difference_still_verifies() -> None:
    """Tier 1 EXACT matching is canonicalization-tolerant (smart quotes,
    whitespace) -- this is still "verbatim" in the sense that matters, and
    must not be rejected over a typographic difference the model didn't
    introduce maliciously."""
    evidence_text = "John’s left arm was lost in the accident."  # curly apostrophe
    fake = FakeStructuredLLMClient(
        default_response={
            "verdict": "SUPPORTED",
            "confidence": 0.9,
            "quotes": ["John's left arm was lost in the accident."],  # straight apostrophe
            "explanation": "x",
        }
    )
    verifier = LLMFactVerifier(fake)

    results = verifier.verify(_claim(), [_fused("chunk-1", evidence_text)])

    assert results[0].label is FactVerificationLabel.SUPPORTED


def test_verify_returns_empty_list_for_empty_evidence() -> None:
    fake = FakeStructuredLLMClient(default_response={})
    verifier = LLMFactVerifier(fake)

    results = verifier.verify(_claim(), [])

    assert results == []
    assert fake.calls == []


def test_verify_returns_exactly_one_result_for_multiple_evidence_records() -> None:
    """The core redesign behavior: one LLM call, one FactVerification,
    regardless of how many evidence records were retrieved (was: one
    result per record, before the redesign)."""
    fake = FakeStructuredLLMClient(
        default_response={"verdict": "NOT_MENTIONED", "confidence": 0.7, "quotes": [], "explanation": "x"}
    )
    verifier = LLMFactVerifier(fake)
    evidence = [_fused(f"chunk-{i}", f"text {i}") for i in range(10)]

    results = verifier.verify(_claim(), evidence)

    assert len(results) == 1
    assert len(fake.calls) == 1


def test_quote_found_in_a_later_evidence_record_anchors_to_that_chunk() -> None:
    """A quote does not have to come from the first evidence record --
    the verifier must check every record in the set, and anchor
    evidence_chunk_id to whichever one actually contains it."""
    fake = FakeStructuredLLMClient(
        default_response={
            "verdict": "CONTRADICTED",
            "confidence": 0.9,
            "quotes": ["John never lost his arm."],
            "explanation": "x",
        }
    )
    verifier = LLMFactVerifier(fake)
    evidence = [
        _fused("chunk-1", "Unrelated passage about the weather."),
        _fused("chunk-2", "Another unrelated passage about a ship."),
        _fused("chunk-3", "John never lost his arm."),
    ]

    results = verifier.verify(_claim(), evidence)

    assert results[0].evidence_chunk_id == "chunk-3"


def test_not_mentioned_anchors_to_the_first_evidence_record_by_rank() -> None:
    """NOT_MENTIONED has no quote to anchor to -- the deterministic
    fallback is the first evidence record by rank, disclosed as a
    representative anchor, not "the" chunk."""
    fake = FakeStructuredLLMClient(
        default_response={"verdict": "NOT_MENTIONED", "confidence": 0.8, "quotes": [], "explanation": "x"}
    )
    verifier = LLMFactVerifier(fake)
    evidence = [_fused("chunk-first", "text a"), _fused("chunk-second", "text b")]

    results = verifier.verify(_claim(), evidence)

    assert results[0].evidence_chunk_id == "chunk-first"


def test_malformed_completion_raises_value_error_not_bare_validation_error() -> None:
    fake = FakeStructuredLLMClient(default_response={"verdict": "NOT_A_REAL_LABEL", "confidence": 0.5, "quotes": [], "explanation": "x"})
    verifier = LLMFactVerifier(fake)

    with pytest.raises(ValueError, match="schema validation"):
        verifier.verify(_claim(), [_fused("chunk-1", "text")])


def test_passes_the_fact_verification_schema_to_the_client() -> None:
    from lncvs.reasoning.fact_verification.llm_schema import FACT_VERIFICATION_JSON_SCHEMA

    fake = FakeStructuredLLMClient(
        default_response={"verdict": "NOT_MENTIONED", "confidence": 0.5, "quotes": [], "explanation": "x"}
    )
    verifier = LLMFactVerifier(fake)

    verifier.verify(_claim(), [_fused("chunk-1", "text")])

    assert len(fake.calls) == 1
    _, schema_used = fake.calls[0]
    assert schema_used == FACT_VERIFICATION_JSON_SCHEMA


def test_prompt_never_contains_a_claim_id_or_parent_claim_id_string() -> None:
    """The verifier renders the prompt from claim.text and the evidence
    records' text only -- AtomicClaim does not even carry the original
    backstory text, so there is no field through which it could leak."""
    fake = FakeStructuredLLMClient(
        default_response={"verdict": "NOT_MENTIONED", "confidence": 0.5, "quotes": [], "explanation": "x"}
    )
    verifier = LLMFactVerifier(fake)
    claim = AtomicClaim(claim_id="claim-abc123", text="John used both hands", parent_claim_id="parent-xyz789")

    verifier.verify(claim, [_fused("chunk-1", "some evidence")])

    prompt_used, _ = fake.calls[0]
    assert "claim-abc123" not in prompt_used
    assert "parent-xyz789" not in prompt_used
