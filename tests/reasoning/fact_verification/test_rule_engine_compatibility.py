"""Backward-compatibility acceptance test (Phase H2): the frozen rule
engine must produce the IDENTICAL FinalVerdict whether it is fed NLIResults
from the existing CrossEncoderNLIVerifier path directly, or from
CrossEncoderFactVerifier -> to_nli_results. Proves the new verifier
abstraction is a true drop-in: classify() and ThresholdRuleEngine are
never modified, never even aware FactVerifier exists.
"""

from lncvs.reasoning.fact_verification import CrossEncoderFactVerifier, to_nli_results
from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.reasoning.nli.service import CrossEncoderNLIVerifier
from lncvs.rules import RuleEngineConfig, ThresholdRuleEngine, classify
from lncvs.schemas import AtomicClaim, EvidenceLedger, FusedEvidence, NLILabel, RetrievalSource, VerdictEnum
from tests.reasoning.nli.fakes import FakeNLIModel


def _claim(claim_id: str, text: str) -> AtomicClaim:
    return AtomicClaim(claim_id=claim_id, text=text)


def _fused(claim_id: str, chunk_id: str, text: str) -> FusedEvidence:
    return FusedEvidence(
        atomic_claim_id=claim_id,
        chunk_id=chunk_id,
        text=text,
        rrf_score=0.5,
        contributing_sources=[RetrievalSource.SEMANTIC],
        contributing_query_ids=["query-1"],
    )


def _ledger_for(claims: list[AtomicClaim]) -> EvidenceLedger:
    ledger = EvidenceLedger(original_claim="multi-fact claim")
    ledger.atomic_claims.extend(claims)
    return ledger


def _config() -> RuleEngineConfig:
    return RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7)


def test_old_and_new_path_produce_identical_verdict_on_contradiction() -> None:
    claim = _claim("claim-1", "John used both hands")
    evidence = [_fused("claim-1", "chunk-arm", "John lost his left arm in 2010.")]
    fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.CONTRADICTION, score=0.93))

    # Old path: CrossEncoderNLIVerifier -> NLIResult directly.
    old_results = CrossEncoderNLIVerifier(fake).verify(claim, evidence)
    old_ledger = _ledger_for([claim])
    old_ledger.nli_results.extend(old_results)
    old_verdict = ThresholdRuleEngine(_config()).evaluate(old_ledger)

    # New path: CrossEncoderFactVerifier -> to_nli_results -> identical NLIResult shape.
    new_fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.CONTRADICTION, score=0.93))
    fact_verifications = CrossEncoderFactVerifier(CrossEncoderNLIVerifier(new_fake)).verify(claim, evidence)
    new_results = to_nli_results(fact_verifications, claim)
    new_ledger = _ledger_for([claim])
    new_ledger.nli_results.extend(new_results)
    new_verdict = ThresholdRuleEngine(_config()).evaluate(new_ledger)

    assert old_verdict.verdict is new_verdict.verdict is VerdictEnum.CONTRADICTORY
    assert old_verdict.verdict == new_verdict.verdict


def test_old_and_new_path_produce_identical_verdict_on_consistent() -> None:
    claim = _claim("claim-1", "John played piano")
    evidence = [_fused("claim-1", "chunk-1", "John played piano in London.")]

    old_fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.ENTAILMENT, score=0.9))
    old_results = CrossEncoderNLIVerifier(old_fake).verify(claim, evidence)
    old_ledger = _ledger_for([claim])
    old_ledger.nli_results.extend(old_results)
    old_verdict = ThresholdRuleEngine(_config()).evaluate(old_ledger)

    new_fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.ENTAILMENT, score=0.9))
    fact_verifications = CrossEncoderFactVerifier(CrossEncoderNLIVerifier(new_fake)).verify(claim, evidence)
    new_results = to_nli_results(fact_verifications, claim)
    new_ledger = _ledger_for([claim])
    new_ledger.nli_results.extend(new_results)
    new_verdict = ThresholdRuleEngine(_config()).evaluate(new_ledger)

    assert old_verdict.verdict == new_verdict.verdict == VerdictEnum.CONSISTENT


def test_classify_produces_identical_contradictions_record_via_either_path() -> None:
    """Also proves the explainability co-product (Contradiction records)
    is unaffected -- classify() itself never changed."""
    claim = _claim("claim-1", "John used both hands")
    evidence = [_fused("claim-1", "chunk-arm", "John lost his left arm in 2010.")]

    old_results = CrossEncoderNLIVerifier(
        FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.CONTRADICTION, score=0.93))
    ).verify(claim, evidence)
    old_outcome = classify(old_results, [claim.claim_id], _config())

    fact_verifications = CrossEncoderFactVerifier(
        CrossEncoderNLIVerifier(FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.CONTRADICTION, score=0.93)))
    ).verify(claim, evidence)
    new_results = to_nli_results(fact_verifications, claim)
    new_outcome = classify(new_results, [claim.claim_id], _config())

    assert len(old_outcome.contradictions) == len(new_outcome.contradictions) == 1
    assert old_outcome.contradictions[0].atomic_claim_id == new_outcome.contradictions[0].atomic_claim_id
    assert old_outcome.contradictions[0].evidence_chunk_id == new_outcome.contradictions[0].evidence_chunk_id
