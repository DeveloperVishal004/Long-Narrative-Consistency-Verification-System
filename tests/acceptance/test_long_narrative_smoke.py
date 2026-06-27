"""Long-Narrative Validation: gated, slow-marked regression guard.

Excluded from the default test run (pyproject.toml: addopts = -m "not slow")
-- select explicitly with `pytest -m slow`. Skips cleanly if the real
embedder, real NLI model, or the narrative file are unavailable in this
environment.

This is Tier 0 only (one claim) -- a fast regression guard proving the
real LangGraph pipeline still executes on the real Castaways narrative
without exception or a fabricated verdict. The full 8-claim Tier 1
experiment with JSON/Markdown reports lives in
scripts/validate_long_narrative.py and is run directly, not via pytest.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lncvs.indexing import EmbeddingConfig, SentenceTransformerEmbedder
from lncvs.reasoning.nli import CrossEncoderNLIModel, NLIConfig
from lncvs.schemas import PipelineStage, VerdictEnum

NARRATIVE_PATH = REPO_ROOT / "data" / "In search of the castaways.txt"
GOLD_DATASET_PATH = REPO_ROOT / "datasets" / "castaways_smoke_claims.jsonl"


@pytest.fixture(scope="module")
def real_embedder() -> SentenceTransformerEmbedder:
    if not NARRATIVE_PATH.is_file():
        pytest.skip(f"Narrative file not found: {NARRATIVE_PATH}")
    try:
        return SentenceTransformerEmbedder(EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Could not load embedding model in this environment: {exc}")


@pytest.fixture(scope="module")
def real_nli_model() -> CrossEncoderNLIModel:
    try:
        return CrossEncoderNLIModel(NLIConfig(model_name="cross-encoder/nli-deberta-v3-base"))
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Could not load NLI model in this environment: {exc}")


@pytest.mark.slow
def test_real_pipeline_executes_on_castaways_without_exception_or_oom(
    real_embedder: SentenceTransformerEmbedder,
    real_nli_model: CrossEncoderNLIModel,
) -> None:
    """Tier 0: one real claim through the real LangGraph pipeline on the
    real ~139k-word Castaways narrative. Must complete, must emit a
    FinalVerdict, must not error."""
    import validate_long_narrative as v

    chunking_config = v.ChunkingConfig(chunk_size=v.CHUNK_SIZE, overlap=v.CHUNK_OVERLAP)

    dataset = v.load_dataset(GOLD_DATASET_PATH, dataset_id="castaways-smoke-gated")
    claim = dataset.examples[0]

    cached_embedder = v.CachingEmbedder(
        real_embedder, v.InMemoryEmbeddingCache(), v.EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    cached_nli = v.CachingNLIModel(
        real_nli_model, v.InMemoryNLICache(), v.NLIConfig(model_name="cross-encoder/nli-deberta-v3-base")
    )

    resources = v.PipelineResources(
        embedder=cached_embedder,
        nli_model=cached_nli,
        decomposition_llm=v.build_decomposition_llm(),
        question_llm=v.FakeLLMClient(default_response="[]"),
        decomposition_config=v.DecompositionConfig(llm_config=v.LLMConfig(model_name="fake-model")),
        question_config=v.QuestionGenerationConfig(llm_config=v.LLMConfig(model_name="fake-model")),
        rule_config=v.RuleEngineConfig(contradiction_threshold=0.5, entailment_threshold=0.5),
        chunking_config=chunking_config,
        retrieval_top_k=10,
    )

    ledger, control, node_latencies = v.run_claim_through_graph(resources, NARRATIVE_PATH, claim.original_claim)

    assert control.current_stage is not PipelineStage.ERROR, f"node failure: {[e.message for e in control.errors]}"
    assert ledger.final_verdict is not None
    assert ledger.final_verdict.verdict in set(VerdictEnum)
    assert len(node_latencies) > 0
