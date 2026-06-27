"""CrossEncoderNLIVerifier tests: evidence-level shape, fixed direction, empty-evidence handling."""

from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.reasoning.nli.service import CrossEncoderNLIVerifier
from lncvs.schemas import AtomicClaim, FusedEvidence, NLILabel, RetrievalSource
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


def test_verify_returns_one_result_per_evidence_record() -> None:
    fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.CONTRADICTION, score=0.9))
    verifier = CrossEncoderNLIVerifier(fake)
    claim = _claim()
    evidence = [_fused("chunk-1", "text one"), _fused("chunk-2", "text two")]

    results = verifier.verify(claim, evidence)

    assert len(results) == 2
    assert {r.evidence_chunk_id for r in results} == {"chunk-1", "chunk-2"}


def test_verify_returns_empty_list_for_empty_evidence() -> None:
    """A claim with zero fused evidence must yield zero NLIResults -- not an
    error, not a fabricated result. This is what ultimately routes the claim
    to INSUFFICIENT_EVIDENCE rather than a false CONTRADICTORY."""
    fake = FakeNLIModel()
    verifier = CrossEncoderNLIVerifier(fake)

    results = verifier.verify(_claim(), [])

    assert results == []
    assert fake.calls == []


def test_verify_fixes_premise_as_evidence_and_hypothesis_as_claim() -> None:
    """Direction-regression test: premise must always be the evidence text,
    hypothesis must always be the atomic claim text. This direction is
    silently reversible and must be pinned by a test, not just code review."""
    fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.CONTRADICTION, score=0.93))
    verifier = CrossEncoderNLIVerifier(fake)
    claim = _claim()
    evidence = [_fused("chunk-arm", "John lost his left arm in an accident in 2010.")]

    results = verifier.verify(claim, evidence)

    assert len(fake.calls) == 1
    called_premise, called_hypothesis = fake.calls[0]
    assert called_premise == "John lost his left arm in an accident in 2010."
    assert called_hypothesis == "John used both hands"

    assert results[0].premise == "John lost his left arm in an accident in 2010."
    assert results[0].hypothesis == "John used both hands"
    assert results[0].atomic_claim_id == "claim-1"
    assert results[0].evidence_chunk_id == "chunk-arm"
    assert results[0].label is NLILabel.CONTRADICTION
    assert results[0].score == 0.93
