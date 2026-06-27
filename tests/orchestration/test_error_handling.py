"""Graph-level error handling: a failing node must record a StageError and
end the run with no fabricated verdict -- never crash, never swallow."""

from pathlib import Path

import pytest

from lncvs.chunking import ChunkingConfig
from lncvs.llm import LLMConfig
from lncvs.orchestration import LangGraphPipeline, PipelineResources
from lncvs.reasoning.decomposition import DecompositionConfig
from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.reasoning.questions import QuestionGenerationConfig
from lncvs.rules import RuleEngineConfig
from lncvs.schemas import AblationVariant, NLILabel, VerdictEnum
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


def _make_pipeline(decomposition_llm: FakeLLMClient) -> LangGraphPipeline:
    resources = PipelineResources(
        embedder=FakeEmbedder(),
        nli_model=SubstringNLIModel(rules=[], default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.5)),
        decomposition_llm=decomposition_llm,
        question_llm=FakeLLMClient(default_response="[]"),
        decomposition_config=DecompositionConfig(llm_config=LLMConfig(model_name="fake-model")),
        question_config=QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
        rule_config=RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7),
        chunking_config=ChunkingConfig(chunk_size=60, overlap=10),
    )
    return LangGraphPipeline(resources)


def test_failing_decomposition_node_raises_with_recorded_stage_error(narrative_path: Path) -> None:
    """FakeLLMClient with no scripted/default response for the rendered
    prompt raises ValueError inside decompose_claim -- the graph must
    surface this as a RuntimeError naming the failed stage, never crash
    uncontrolled and never produce a fabricated verdict."""
    broken_llm = FakeLLMClient(scripted={})  # no default_response -> raises on any prompt
    pipeline = _make_pipeline(broken_llm)

    with pytest.raises(RuntimeError, match="CLAIM_DECOMPOSITION"):
        pipeline.run(narrative_path, ORIGINAL_CLAIM, AblationVariant(name="full"))


def test_missing_evidence_claim_routes_to_insufficient_evidence_not_error(narrative_path: Path) -> None:
    """A claim with zero retrieved/fused evidence is NOT a node failure --
    it is the correct input to INSUFFICIENT_EVIDENCE. This is the cardinal
    invariant: a coverage gap must never silently become a crash or a
    fabricated CONTRADICTORY."""
    pipeline = _make_pipeline(FakeLLMClient(default_response=DECOMPOSITION_RESPONSE))

    ledger = pipeline.run(narrative_path, ORIGINAL_CLAIM, AblationVariant(name="full"))

    assert ledger.final_verdict is not None
    assert ledger.final_verdict.verdict is VerdictEnum.INSUFFICIENT_EVIDENCE
