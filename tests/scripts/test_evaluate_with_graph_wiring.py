"""Phase H4 wiring tests: build_fact_verifier's dependency injection and
configuration switching, plus an end-to-end proof that run_claim produces
correct verdicts through EITHER FactVerifier implementation -- with zero
branching in run_claim itself. All offline: FakeEmbedder (no model
download), FakeNLIModel, FakeStructuredLLMClient, FakeLLMClient. No API
calls, no real Chroma/cross-encoder model loads.
"""

import pytest

from evaluate_with_graph import build_fact_verifier, run_claim
from lncvs.indexing import BM25Index, ChromaIndex
from lncvs.llm import LLMConfig
from lncvs.reasoning.decomposition import DecompositionConfig
from lncvs.reasoning.fact_verification import CrossEncoderFactVerifier, LLMFactVerifier
from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.rules import RuleEngineConfig
from lncvs.schemas import DocumentChunk, NLILabel, VerdictEnum
from tests.indexing.fakes import FakeEmbedder
from tests.llm.fakes import FakeLLMClient, FakeStructuredLLMClient
from tests.reasoning.nli.fakes import FakeNLIModel

CONTRADICTING_CHUNK = DocumentChunk(
    chunk_id="chunk-arm", text="John lost his left arm in an accident in 2010.", char_start=0, char_end=48, source_id="demo"
)
NEUTRAL_CHUNK = DocumentChunk(
    chunk_id="chunk-london", text="John moved to London in 2012.", char_start=49, char_end=79, source_id="demo"
)


def _indices(chunks: list[DocumentChunk] | None = None) -> tuple[ChromaIndex, BM25Index]:
    chunks = chunks if chunks is not None else [CONTRADICTING_CHUNK, NEUTRAL_CHUNK]
    chroma = ChromaIndex(embedder=FakeEmbedder(), collection_name=f"h4-wiring-test-{id(chunks)}")
    chroma.index(chunks)
    bm25 = BM25Index(collection_name=f"h4-wiring-test-bm25-{id(chunks)}")
    bm25.index(chunks)
    return chroma, bm25


def _decomposition_llm(claim_text: str) -> FakeLLMClient:
    return FakeLLMClient(default_response=f'["{claim_text}"]')


def _decomp_config() -> DecompositionConfig:
    return DecompositionConfig(llm_config=LLMConfig(model_name="fake-model"))


def _rule_config() -> RuleEngineConfig:
    return RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7, consistency_requires_entailment=False)


# --- build_fact_verifier: dependency injection + configuration switching ---


def test_build_fact_verifier_cross_encoder_mode_returns_cross_encoder_fact_verifier() -> None:
    verifier = build_fact_verifier("cross_encoder", FakeNLIModel())
    assert isinstance(verifier, CrossEncoderFactVerifier)


def test_build_fact_verifier_llm_mode_returns_llm_fact_verifier() -> None:
    verifier = build_fact_verifier("llm", None)
    assert isinstance(verifier, LLMFactVerifier)


def test_build_fact_verifier_cross_encoder_mode_requires_a_model() -> None:
    with pytest.raises(ValueError, match="requires a cached_nli_model"):
        build_fact_verifier("cross_encoder", None)


def test_build_fact_verifier_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown VERIFIER_MODE"):
        build_fact_verifier("not_a_real_mode", None)


# --- run_claim: identical downstream behavior through EITHER verifier ---


def test_run_claim_resolves_contradictory_via_cross_encoder_verifier() -> None:
    chroma, bm25 = _indices()
    claim_text = "John used both hands"
    fake_nli = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.CONTRADICTION, score=0.93))
    fact_verifier = build_fact_verifier("cross_encoder", fake_nli)

    ledger, _ = run_claim(claim_text, chroma, bm25, None, _decomposition_llm(claim_text), _decomp_config(), fact_verifier, _rule_config())

    assert ledger.final_verdict.verdict is VerdictEnum.CONTRADICTORY


def test_run_claim_resolves_contradictory_via_llm_verifier() -> None:
    """Identical scenario to the cross-encoder test above, swapping ONLY
    the injected FactVerifier -- proves run_claim is genuinely unaware of
    which implementation it received. Single-chunk index: the fake LLM
    client scripts one quote, and LLMFactVerifier correctly rejects that
    quote against any OTHER chunk it doesn't appear in (the trust-boundary
    check working as intended) -- a multi-chunk index would otherwise
    raise on the second, unrelated chunk."""
    chroma, bm25 = _indices([CONTRADICTING_CHUNK])
    claim_text = "John used both hands"
    fake_structured = FakeStructuredLLMClient(
        default_response={
            "verdict": "CONTRADICTED",
            "confidence": 0.93,
            "quotes": ["John lost his left arm in an accident in 2010."],
            "explanation": "Direct contradiction.",
        }
    )
    fact_verifier = LLMFactVerifier(fake_structured)

    ledger, _ = run_claim(claim_text, chroma, bm25, None, _decomposition_llm(claim_text), _decomp_config(), fact_verifier, _rule_config())

    assert ledger.final_verdict.verdict is VerdictEnum.CONTRADICTORY


def test_run_claim_resolves_consistent_via_cross_encoder_verifier() -> None:
    chroma, bm25 = _indices()
    claim_text = "John moved to London"
    fake_nli = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.6))
    fact_verifier = build_fact_verifier("cross_encoder", fake_nli)

    ledger, _ = run_claim(claim_text, chroma, bm25, None, _decomposition_llm(claim_text), _decomp_config(), fact_verifier, _rule_config())

    assert ledger.final_verdict.verdict is VerdictEnum.CONSISTENT


def test_run_claim_resolves_consistent_via_llm_verifier() -> None:
    chroma, bm25 = _indices()
    claim_text = "John moved to London"
    fake_structured = FakeStructuredLLMClient(
        default_response={
            "verdict": "NOT_MENTIONED",
            "confidence": 0.7,
            "quotes": [],
            "explanation": "Unrelated to the retrieved evidence.",
        }
    )
    fact_verifier = LLMFactVerifier(fake_structured)

    ledger, _ = run_claim(claim_text, chroma, bm25, None, _decomposition_llm(claim_text), _decomp_config(), fact_verifier, _rule_config())

    assert ledger.final_verdict.verdict is VerdictEnum.CONSISTENT


def test_run_claim_ledger_upstream_fields_unaffected_by_verifier_choice() -> None:
    """The Ledger's UPSTREAM-of-verification fields (atomic_claims,
    retrieved_evidence, fused_evidence) are populated identically in shape
    regardless of which FactVerifier produced the eventual NLIResults --
    proves the Ledger and everything upstream of verification are
    genuinely unmodified by Phase H4/the evidence-set-level redesign.

    nli_results count is explicitly NOT asserted equal here -- that is the
    deliberate, documented difference the redesign introduces:
    CrossEncoderFactVerifier remains evidence-level (one NLIResult per
    fused evidence record), while LLMFactVerifier is now evidence-SET-level
    (exactly one NLIResult per claim, regardless of evidence count)."""
    chroma, bm25 = _indices()
    claim_text = "John used both hands"

    cross_encoder_verifier = build_fact_verifier("cross_encoder", FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.5)))
    llm_verifier = LLMFactVerifier(
        FakeStructuredLLMClient(default_response={"verdict": "NOT_MENTIONED", "confidence": 0.5, "quotes": [], "explanation": "x"})
    )

    ledger_a, _ = run_claim(claim_text, chroma, bm25, None, _decomposition_llm(claim_text), _decomp_config(), cross_encoder_verifier, _rule_config())
    ledger_b, _ = run_claim(claim_text, chroma, bm25, None, _decomposition_llm(claim_text), _decomp_config(), llm_verifier, _rule_config())

    assert len(ledger_a.atomic_claims) == len(ledger_b.atomic_claims) == 1
    assert len(ledger_a.retrieved_evidence) == len(ledger_b.retrieved_evidence)
    assert len(ledger_a.fused_evidence) == len(ledger_b.fused_evidence)

    # The redesign's defining behavioral difference, asserted explicitly:
    assert len(ledger_a.nli_results) == len(ledger_a.fused_evidence)  # cross-encoder: evidence-level
    assert len(ledger_b.nli_results) == 1  # LLM: evidence-set-level, always exactly 1
