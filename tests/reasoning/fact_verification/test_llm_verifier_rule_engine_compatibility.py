"""Proves LLMFactVerifier is ALSO a true drop-in for the frozen rule
engine -- mirrors test_rule_engine_compatibility.py's CrossEncoderFactVerifier
proof exactly, swapping in LLMFactVerifier. classify() and
ThresholdRuleEngine are never modified, never even aware which FactVerifier
implementation produced the NLIResults they consume."""

from lncvs.reasoning.fact_verification import LLMFactVerifier, to_nli_results
from lncvs.rules import RuleEngineConfig, ThresholdRuleEngine
from lncvs.schemas import AtomicClaim, EvidenceLedger, FusedEvidence, RetrievalSource, VerdictEnum
from tests.llm.fakes import FakeStructuredLLMClient


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


def _config() -> RuleEngineConfig:
    return RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7)


def test_llm_fact_verifier_produces_contradictory_verdict_via_frozen_rule_engine() -> None:
    claim = _claim("claim-1", "John used both hands")
    evidence_text = "John lost his left arm in 2010."
    evidence = [_fused("claim-1", "chunk-arm", evidence_text)]

    fake = FakeStructuredLLMClient(
        default_response={
            "verdict": "CONTRADICTED",
            "confidence": 0.93,
            "quotes": ["John lost his left arm in 2010."],
            "explanation": "Direct contradiction.",
        }
    )
    fact_verifications = LLMFactVerifier(fake).verify(claim, evidence)
    nli_results = to_nli_results(fact_verifications, claim)

    ledger = EvidenceLedger(original_claim="multi-fact claim")
    ledger.atomic_claims.append(claim)
    ledger.nli_results.extend(nli_results)
    verdict = ThresholdRuleEngine(_config()).evaluate(ledger)

    assert verdict.verdict is VerdictEnum.CONTRADICTORY


def test_llm_fact_verifier_not_mentioned_routes_to_insufficient_evidence() -> None:
    """The cardinal invariant, now exercised through the LLM verifier path:
    NOT_MENTIONED (-> NEUTRAL) must never be mistaken for a contradiction."""
    claim = _claim("claim-1", "John used both hands")
    evidence = [_fused("claim-1", "chunk-1", "John moved to Paris in 1820.")]

    fake = FakeStructuredLLMClient(
        default_response={
            "verdict": "NOT_MENTIONED",
            "confidence": 0.8,
            "quotes": [],
            "explanation": "Unrelated passage.",
        }
    )
    fact_verifications = LLMFactVerifier(fake).verify(claim, evidence)
    nli_results = to_nli_results(fact_verifications, claim)

    ledger = EvidenceLedger(original_claim="multi-fact claim")
    ledger.atomic_claims.append(claim)
    ledger.nli_results.extend(nli_results)
    verdict = ThresholdRuleEngine(_config()).evaluate(ledger)

    assert verdict.verdict is VerdictEnum.INSUFFICIENT_EVIDENCE


def test_llm_fact_verifier_supported_routes_to_consistent() -> None:
    claim = _claim("claim-1", "John played piano")
    evidence = [_fused("claim-1", "chunk-1", "John played piano in London.")]

    fake = FakeStructuredLLMClient(
        default_response={
            "verdict": "SUPPORTED",
            "confidence": 0.9,
            "quotes": ["John played piano in London."],
            "explanation": "Direct match.",
        }
    )
    fact_verifications = LLMFactVerifier(fake).verify(claim, evidence)
    nli_results = to_nli_results(fact_verifications, claim)

    ledger = EvidenceLedger(original_claim="multi-fact claim")
    ledger.atomic_claims.append(claim)
    ledger.nli_results.extend(nli_results)
    verdict = ThresholdRuleEngine(_config()).evaluate(ledger)

    assert verdict.verdict is VerdictEnum.CONSISTENT
