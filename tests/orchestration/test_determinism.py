"""N repeated LangGraphPipeline.run() calls on identical input must produce
identical ledger_fingerprint -- the same determinism guarantee every other
phase's runner is held to."""

from pathlib import Path

import pytest

from lncvs.chunking import ChunkingConfig
from lncvs.evaluation import ledger_fingerprint
from lncvs.llm import LLMConfig
from lncvs.orchestration import LangGraphPipeline, PipelineResources
from lncvs.reasoning.decomposition import DecompositionConfig
from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.reasoning.questions import QuestionGenerationConfig
from lncvs.rules import RuleEngineConfig
from lncvs.schemas import AblationVariant, NLILabel
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


def _make_pipeline() -> LangGraphPipeline:
    resources = PipelineResources(
        embedder=FakeEmbedder(),
        nli_model=SubstringNLIModel(
            rules=[("lost his left arm", "used both hands", NLIPrediction(label=NLILabel.CONTRADICTION, score=0.95))],
            default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.5),
        ),
        decomposition_llm=FakeLLMClient(default_response=DECOMPOSITION_RESPONSE),
        question_llm=FakeLLMClient(default_response="[]"),
        decomposition_config=DecompositionConfig(llm_config=LLMConfig(model_name="fake-model")),
        question_config=QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
        rule_config=RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7),
        chunking_config=ChunkingConfig(chunk_size=60, overlap=10),
    )
    return LangGraphPipeline(resources)


def test_graph_run_is_deterministic_across_repeated_invocations(narrative_path: Path) -> None:
    def run_once() -> str:
        ledger = _make_pipeline().run(narrative_path, ORIGINAL_CLAIM, AblationVariant(name="full"))
        return ledger_fingerprint(ledger)

    fingerprints = {run_once() for _ in range(3)}
    assert len(fingerprints) == 1
