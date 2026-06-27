"""The Phase 7 acceptance gate: LangGraphPipeline must produce a
ledger_fingerprint-identical EvidenceLedger to PipelineRunner, for every
standard ablation variant, given the same injected fakes and configs.

This is what makes "preserve behavior" a structural guarantee: both runners
call the identical underlying functions (build_retrieval_queries,
fuse_evidence/round_robin_fuse, classify, ThresholdRuleEngine.evaluate, the
LedgerService.record_*() methods) in the same order. If a future change to
either runner drifts from the other, this test is what catches it.
"""

from pathlib import Path

import pytest

from lncvs.chunking import ChunkingConfig
from lncvs.evaluation import PipelineRunner, ledger_fingerprint, standard_ablation_matrix
from lncvs.llm import LLMConfig
from lncvs.orchestration import LangGraphPipeline, PipelineResources
from lncvs.reasoning.decomposition import DecompositionConfig
from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.reasoning.questions import QuestionGenerationConfig
from lncvs.rules import RuleEngineConfig
from lncvs.schemas import NLILabel, VerdictEnum
from tests.evaluation.fakes import SubstringNLIModel
from tests.indexing.fakes import FakeEmbedder
from tests.llm.fakes import FakeLLMClient

NARRATIVE_TEXT = "John lost his left arm in an accident in 2010.\n\nJohn moved to London in 2012.\n"
ORIGINAL_CLAIM = "John played a two-handed piano piece in London."
DECOMPOSITION_RESPONSE = '["John played piano", "John used both hands", "the event occurred in London"]'


@pytest.fixture()
def narrative_path(tmp_path: Path) -> Path:
    path = tmp_path / "narrative.txt"
    path.write_text(NARRATIVE_TEXT, encoding="utf-8")
    return path


def _nli_model():
    return SubstringNLIModel(
        rules=[("lost his left arm", "used both hands", NLIPrediction(label=NLILabel.CONTRADICTION, score=0.95))],
        default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.5),
    )


def _common_kwargs() -> dict:
    return dict(
        decomposition_config=DecompositionConfig(llm_config=LLMConfig(model_name="fake-model")),
        question_config=QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
        rule_config=RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7),
        chunking_config=ChunkingConfig(chunk_size=60, overlap=10),
    )


def _make_runner() -> PipelineRunner:
    return PipelineRunner(
        embedder=FakeEmbedder(),
        nli_model=_nli_model(),
        decomposition_llm=FakeLLMClient(default_response=DECOMPOSITION_RESPONSE),
        question_llm=FakeLLMClient(default_response="[]"),
        **_common_kwargs(),
    )


def _make_pipeline() -> LangGraphPipeline:
    resources = PipelineResources(
        embedder=FakeEmbedder(),
        nli_model=_nli_model(),
        decomposition_llm=FakeLLMClient(default_response=DECOMPOSITION_RESPONSE),
        question_llm=FakeLLMClient(default_response="[]"),
        **_common_kwargs(),
    )
    return LangGraphPipeline(resources)


@pytest.mark.parametrize("variant", standard_ablation_matrix(), ids=lambda v: v.name)
def test_graph_matches_runner_fingerprint_for_every_ablation_variant(narrative_path: Path, variant) -> None:
    runner_ledger = _make_runner().run(narrative_path, ORIGINAL_CLAIM, variant)
    graph_ledger = _make_pipeline().run(narrative_path, ORIGINAL_CLAIM, variant)

    assert ledger_fingerprint(runner_ledger) == ledger_fingerprint(graph_ledger)
    assert runner_ledger.final_verdict.verdict == graph_ledger.final_verdict.verdict
    assert runner_ledger.final_verdict.fired_rule == graph_ledger.final_verdict.fired_rule


def test_graph_matches_runner_on_section_14_dummy_case(narrative_path: Path) -> None:
    """The standing PROJECT_SPEC.md Section 14 acceptance case, through both runners."""
    from lncvs.schemas import standard_ablation_matrix as sam

    full_variant = next(v for v in sam() if v.name == "full")

    runner_ledger = _make_runner().run(narrative_path, ORIGINAL_CLAIM, full_variant)
    graph_ledger = _make_pipeline().run(narrative_path, ORIGINAL_CLAIM, full_variant)

    assert runner_ledger.final_verdict.verdict is VerdictEnum.CONTRADICTORY
    assert graph_ledger.final_verdict.verdict is VerdictEnum.CONTRADICTORY
    assert ledger_fingerprint(runner_ledger) == ledger_fingerprint(graph_ledger)
