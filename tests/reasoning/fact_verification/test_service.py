"""CrossEncoderFactVerifier tests: protocol conformance, evidence-level
shape, label re-mapping, and that it changes no NLI inference behavior --
only wraps and re-labels the existing CrossEncoderNLIVerifier."""

from lncvs.reasoning.fact_verification import CrossEncoderFactVerifier, FactVerifier
from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.reasoning.nli.service import CrossEncoderNLIVerifier
from lncvs.schemas import AtomicClaim, FactVerificationLabel, FusedEvidence, NLILabel, RetrievalSource
from tests.reasoning.nli.fakes import FakeNLIModel


def _claim() -> AtomicClaim:
    return AtomicClaim(claim_id="claim-1", text="John used both hands")


def _fused(chunk_id: str, text: str) -> FusedEvidence:
    return FusedEvidence(
        atomic_claim_id="claim-1",
        chunk_id=chunk_id,
        text=text,
        rrf_score=0.5,
        contributing_sources=[RetrievalSource.SEMANTIC],
        contributing_query_ids=["query-1"],
    )


def test_cross_encoder_fact_verifier_satisfies_fact_verifier_protocol() -> None:
    verifier = CrossEncoderFactVerifier(CrossEncoderNLIVerifier(FakeNLIModel()))
    assert isinstance(verifier, FactVerifier)


def test_verify_returns_one_fact_verification_per_evidence_record() -> None:
    fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.CONTRADICTION, score=0.9))
    verifier = CrossEncoderFactVerifier(CrossEncoderNLIVerifier(fake))
    claim = _claim()
    evidence = [_fused("chunk-1", "text one"), _fused("chunk-2", "text two")]

    results = verifier.verify(claim, evidence)

    assert len(results) == 2
    assert {r.evidence_chunk_id for r in results} == {"chunk-1", "chunk-2"}


def test_verify_returns_empty_list_for_empty_evidence() -> None:
    verifier = CrossEncoderFactVerifier(CrossEncoderNLIVerifier(FakeNLIModel()))

    results = verifier.verify(_claim(), [])

    assert results == []


def test_entailment_maps_to_supported() -> None:
    fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.ENTAILMENT, score=0.95))
    verifier = CrossEncoderFactVerifier(CrossEncoderNLIVerifier(fake))

    results = verifier.verify(_claim(), [_fused("chunk-1", "John used both hands to play.")])

    assert results[0].label is FactVerificationLabel.SUPPORTED
    assert results[0].confidence == 0.95


def test_contradiction_maps_to_contradicted() -> None:
    fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.CONTRADICTION, score=0.93))
    verifier = CrossEncoderFactVerifier(CrossEncoderNLIVerifier(fake))

    results = verifier.verify(_claim(), [_fused("chunk-1", "John lost his left arm in 2010.")])

    assert results[0].label is FactVerificationLabel.CONTRADICTED
    assert results[0].confidence == 0.93


def test_neutral_maps_to_not_mentioned() -> None:
    fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.7))
    verifier = CrossEncoderFactVerifier(CrossEncoderNLIVerifier(fake))

    results = verifier.verify(_claim(), [_fused("chunk-1", "John moved to Paris.")])

    assert results[0].label is FactVerificationLabel.NOT_MENTIONED
    assert results[0].confidence == 0.7


def test_supporting_quotes_falls_back_to_full_evidence_text() -> None:
    fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.CONTRADICTION, score=0.9))
    verifier = CrossEncoderFactVerifier(CrossEncoderNLIVerifier(fake))
    evidence_text = "John lost his left arm in an accident in 2010."

    results = verifier.verify(_claim(), [_fused("chunk-arm", evidence_text)])

    assert results[0].supporting_quotes == (evidence_text,)


def test_explanation_is_non_empty_and_mentions_the_label() -> None:
    fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.ENTAILMENT, score=0.8))
    verifier = CrossEncoderFactVerifier(CrossEncoderNLIVerifier(fake))

    results = verifier.verify(_claim(), [_fused("chunk-1", "text")])

    assert "ENTAILMENT" in results[0].explanation


def test_atomic_claim_id_and_evidence_chunk_id_are_preserved() -> None:
    fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.5))
    verifier = CrossEncoderFactVerifier(CrossEncoderNLIVerifier(fake))

    results = verifier.verify(_claim(), [_fused("chunk-99", "text")])

    assert results[0].atomic_claim_id == "claim-1"
    assert results[0].evidence_chunk_id == "chunk-99"
