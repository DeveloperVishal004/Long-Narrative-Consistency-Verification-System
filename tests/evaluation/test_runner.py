"""PipelineRunner tests: offline (FakeEmbedder/FakeLLMClient/FakeNLIModel, real ChromaIndex/BM25Index),
exercising every AblationVariant toggle.
"""

from pathlib import Path

import pytest

from lncvs.chunking import ChunkingConfig
from lncvs.evaluation import AblationVariant, FusionStrategy, PipelineRunner
from lncvs.llm import LLMConfig
from lncvs.reasoning.decomposition import DecompositionConfig
from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.reasoning.questions import QuestionGenerationConfig
from lncvs.rules import RuleEngineConfig
from lncvs.schemas import NLILabel, RetrievalSource, VerdictEnum
from tests.evaluation.fakes import SubstringNLIModel
from tests.indexing.fakes import FakeEmbedder
from tests.llm.fakes import FakeLLMClient

ORIGINAL_CLAIM = "John played a two-handed piano piece in London."
DECOMPOSITION_RESPONSE = '["John played piano", "John used both hands", "the event occurred in London"]'


@pytest.fixture()
def narrative_path(tmp_path: Path) -> Path:
    path = tmp_path / "narrative.txt"
    path.write_text(
        "John lost his left arm in an accident in 2010.\n\nJohn moved to London in 2012.\n", encoding="utf-8"
    )
    return path


def _make_runner() -> PipelineRunner:
    nli_model = SubstringNLIModel(
        rules=[("lost his left arm", "used both hands", NLIPrediction(label=NLILabel.CONTRADICTION, score=0.95))],
        default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.5),
    )
    return PipelineRunner(
        embedder=FakeEmbedder(),
        nli_model=nli_model,
        decomposition_llm=FakeLLMClient(default_response=DECOMPOSITION_RESPONSE),
        question_llm=FakeLLMClient(default_response="[]"),
        decomposition_config=DecompositionConfig(llm_config=LLMConfig(model_name="fake-model")),
        question_config=QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
        rule_config=RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7),
        chunking_config=ChunkingConfig(chunk_size=60, overlap=10),
    )


def test_run_returns_ledger_with_final_verdict(narrative_path: Path) -> None:
    runner = _make_runner()
    variant = AblationVariant(name="full")

    ledger = runner.run(narrative_path, ORIGINAL_CLAIM, variant)

    assert ledger.final_verdict is not None
    assert len(ledger.atomic_claims) == 3


def test_no_question_generation_variant_yields_zero_probe_questions(narrative_path: Path) -> None:
    runner = _make_runner()
    variant = AblationVariant(name="no_question_generation", use_question_generation=False)

    ledger = runner.run(narrative_path, ORIGINAL_CLAIM, variant)

    assert ledger.probe_questions == []


def test_no_bm25_variant_yields_semantic_only_evidence(narrative_path: Path) -> None:
    runner = _make_runner()
    variant = AblationVariant(name="no_bm25", use_bm25=False)

    ledger = runner.run(narrative_path, ORIGINAL_CLAIM, variant)

    sources = {evidence.source for evidence in ledger.retrieved_evidence}
    assert sources <= {RetrievalSource.SEMANTIC}


def test_round_robin_variant_still_produces_fused_evidence(narrative_path: Path) -> None:
    runner = _make_runner()
    variant = AblationVariant(name="no_rrf", fusion_strategy=FusionStrategy.ROUND_ROBIN)

    ledger = runner.run(narrative_path, ORIGINAL_CLAIM, variant)

    assert len(ledger.fused_evidence) > 0
    assert ledger.final_verdict is not None


def test_runner_is_deterministic_across_repeated_runs(narrative_path: Path, tmp_path: Path) -> None:
    def run_once() -> tuple[VerdictEnum, str]:
        runner = _make_runner()
        variant = AblationVariant(name="full")
        ledger = runner.run(narrative_path, ORIGINAL_CLAIM, variant)
        return ledger.final_verdict.verdict, ledger.final_verdict.fired_rule

    assert run_once() == run_once()


def test_dummy_case_resolves_to_contradictory_under_full_variant(narrative_path: Path) -> None:
    runner = _make_runner()
    variant = AblationVariant(name="full")

    ledger = runner.run(narrative_path, ORIGINAL_CLAIM, variant)

    assert ledger.final_verdict.verdict is VerdictEnum.CONTRADICTORY
