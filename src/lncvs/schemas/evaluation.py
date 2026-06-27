"""Evaluation framework contracts: gold datasets, metrics, and lightweight reports.

EvaluationReport and ExampleResult are deliberately lightweight: they store
fingerprints and aggregated metrics, never an embedded EvidenceLedger. This
keeps evaluation reports cheap to persist and compare across runs, per the
Phase 6 architecture review's explicit "lightweight references/fingerprints"
requirement -- full ledgers, if needed for debugging, are written separately
to disk and referenced only by ExampleResult.ledger_path.

AblationVariant IS defined here (not in evaluation/config.py, where it
originated in Phase 6): Phase 7's orchestration/ also needs it to decide a
graph node's control flow (use_bm25, fusion_strategy), and orchestration/
must not import from evaluation/ -- both packages import from this leaf
instead. evaluation/config.py re-exports it for backward compatibility, so
no Phase 6 import site changed. EvaluationReport itself still never embeds
an AblationVariant -- it stores the variant's name and fingerprint as plain
strings, which is the actual reason this module stays a leaf: nothing here
depends on EvaluationReport or vice versa in a way that would invert
anything.
"""

import hashlib
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lncvs.schemas.enums import FusionStrategy, PipelineStage, VerdictEnum

_ALL_VERDICTS = list(VerdictEnum)


class AblationVariant(BaseModel):
    """One point in the ablation matrix: which pipeline components are enabled.

    Consumed by both evaluation/ (to build the ablation matrix) and
    orchestration/ (PipelineRunner and, from Phase 7, LangGraphPipeline read
    use_question_generation/use_bm25/fusion_strategy to decide control flow
    per run) -- the reason this type lives in schemas/, the one module both
    can import without creating a cycle.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, description="Human-readable label for this variant.")
    use_question_generation: bool = Field(default=True)
    use_bm25: bool = Field(default=True)
    fusion_strategy: FusionStrategy = Field(default=FusionStrategy.RRF)

    def fingerprint(self) -> str:
        """Deterministic identifier for this exact variant's settings."""
        digest_input = (
            f"{self.use_question_generation}:{self.use_bm25}:{self.fusion_strategy.value}"
        ).encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()[:16]


def standard_ablation_matrix() -> list[AblationVariant]:
    """The four standard variants: the full pipeline, and one component removed at a time."""
    return [
        AblationVariant(name="full"),
        AblationVariant(name="no_question_generation", use_question_generation=False),
        AblationVariant(name="no_bm25", use_bm25=False),
        AblationVariant(name="no_rrf", fusion_strategy=FusionStrategy.ROUND_ROBIN),
    ]


class GoldSpan(BaseModel):
    """A character span within a narrative's cleaned_text marking gold-relevant evidence.

    Span-based, not chunk-id-based: chunk_id is a content hash that changes
    whenever chunking config changes, so chunk-id gold labels would silently
    break under re-chunking. Spans are mapped to whichever chunks currently
    cover them at evaluation time (see lncvs.evaluation.dataset.map_spans_to_chunks).
    """

    model_config = ConfigDict(frozen=True)

    char_start: int = Field(..., ge=0, description="Start offset within the narrative's cleaned_text.")
    char_end: int = Field(..., ge=0, description="End offset within the narrative's cleaned_text.")
    note: str | None = Field(default=None, description="Optional human-readable annotation.")

    @model_validator(mode="after")
    def _validate_span(self) -> "GoldSpan":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be strictly greater than char_start")
        return self


class GoldExample(BaseModel):
    """A single (narrative, claim) pair with its expected verdict and gold evidence spans."""

    model_config = ConfigDict(frozen=True)

    example_id: str = Field(..., min_length=1, description="Unique identifier for this gold example.")
    narrative_path: str = Field(..., min_length=1, description="Filesystem path to the source narrative.")
    original_claim: str = Field(..., min_length=1, description="The claim to verify.")
    expected_verdict: VerdictEnum = Field(..., description="The gold verdict for this example.")
    gold_evidence: list[GoldSpan] = Field(
        default_factory=list, description="Spans marking evidence relevant to this claim, of any polarity."
    )
    gold_contradicting_spans: list[GoldSpan] = Field(
        default_factory=list, description="Spans marking evidence that specifically contradicts the claim."
    )


class EvaluationDataset(BaseModel):
    """A named collection of gold examples."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(..., min_length=1, description="Unique identifier for this dataset.")
    examples: list[GoldExample] = Field(..., min_length=1, description="The gold examples in this dataset.")

    def fingerprint(self) -> str:
        """Deterministic identifier for this exact dataset's content."""
        digest_input = self.model_dump_json().encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()[:16]


class RankCutoffMetric(BaseModel):
    """Recall and precision at a single rank cutoff k."""

    model_config = ConfigDict(frozen=True)

    k: int = Field(..., gt=0)
    recall: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)


class RetrievalMetrics(BaseModel):
    """Retrieval/fusion quality: MRR plus Recall@k/Precision@k at each configured cutoff."""

    model_config = ConfigDict(frozen=True)

    mrr: float = Field(..., ge=0.0, le=1.0)
    cutoffs: list[RankCutoffMetric] = Field(..., min_length=1)


class PerClassMetric(BaseModel):
    """Precision/recall/F1/support for a single VerdictEnum class."""

    model_config = ConfigDict(frozen=True)

    verdict: VerdictEnum
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    support: int = Field(..., ge=0, description="Number of gold examples with this verdict.")


class ConfusionCell(BaseModel):
    """A single (gold, predicted) cell of the verdict confusion matrix."""

    model_config = ConfigDict(frozen=True)

    gold: VerdictEnum
    predicted: VerdictEnum
    count: int = Field(..., ge=0)


class ConfusionMatrix(BaseModel):
    """Typed 3x3 confusion matrix over VerdictEnum, expressed as sparse non-zero cells."""

    model_config = ConfigDict(frozen=True)

    cells: list[ConfusionCell] = Field(..., min_length=1)


class VerdictMetrics(BaseModel):
    """Verdict-quality metrics, strictly 3-class over VerdictEnum.

    contradiction_detection_rate is recall on the CONTRADICTORY class
    specifically; it is None when no gold-CONTRADICTORY examples exist in
    the dataset, never silently reported as 0.
    """

    model_config = ConfigDict(frozen=True)

    accuracy: float = Field(..., ge=0.0, le=1.0)
    macro_precision: float = Field(..., ge=0.0, le=1.0)
    macro_recall: float = Field(..., ge=0.0, le=1.0)
    macro_f1: float = Field(..., ge=0.0, le=1.0)
    per_class: list[PerClassMetric] = Field(..., min_length=1)
    confusion: ConfusionMatrix
    contradiction_detection_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class CitationMetrics(BaseModel):
    """Citation-quality metrics over the evidence actually cited in contradictions/supporting_evidence."""

    model_config = ConfigDict(frozen=True)

    citation_accuracy: float = Field(..., ge=0.0, le=1.0)
    hallucination_rate: float = Field(..., ge=0.0, le=1.0)
    cited_count: int = Field(..., ge=0)
    grounded_count: int = Field(..., ge=0)


class StageLatency(BaseModel):
    """Duration of a single pipeline stage, derived from ledger_log timestamps."""

    model_config = ConfigDict(frozen=True)

    stage: PipelineStage
    duration_ms: float = Field(..., ge=0.0)


class LatencyMetrics(BaseModel):
    """Latency is the one metric NOT subject to the determinism guarantee.

    Per CLAUDE.md, wall-clock fields (ledger_log timestamps) are an accepted
    exception to full ledger reproducibility -- the rule engine never reads
    them, so they may differ between runs even on identical input.
    """

    model_config = ConfigDict(frozen=True)

    stages: list[StageLatency] = Field(default_factory=list)
    end_to_end_ms: float = Field(..., ge=0.0)


class ProvenanceFingerprints(BaseModel):
    """Config fingerprints recorded for audit: given these plus the dataset, a report is independently re-derivable."""

    model_config = ConfigDict(frozen=True)

    embedding_fp: str | None = Field(default=None)
    nli_fp: str | None = Field(default=None)
    decomposition_llm_fp: str | None = Field(default=None)
    question_llm_fp: str | None = Field(default=None)
    fusion_fp: str | None = Field(default=None)
    rule_engine_fp: str | None = Field(default=None)
    eval_config_fp: str = Field(..., min_length=1)
    seed: int = Field(...)


class ExampleResult(BaseModel):
    """Lightweight, per-example evaluation outcome. Never embeds a full EvidenceLedger."""

    model_config = ConfigDict(frozen=True)

    example_id: str = Field(..., min_length=1)
    predicted_verdict: VerdictEnum
    expected_verdict: VerdictEnum
    fired_rule: str = Field(..., min_length=1)
    correct: bool
    retrieval: RetrievalMetrics | None = Field(default=None)
    citation: CitationMetrics | None = Field(default=None)
    latency: LatencyMetrics
    ledger_fingerprint: str = Field(..., min_length=1)
    ledger_path: str | None = Field(
        default=None, description="Optional path to a persisted ledger, if EvaluationConfig.persist_ledgers is set."
    )


class EvaluationReport(BaseModel):
    """The lightweight output of one EvaluationHarness.evaluate_variant() call."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., min_length=1, description="Deterministic hash of (dataset, variant, config) fingerprints.")
    variant_name: str = Field(..., min_length=1)
    variant_fingerprint: str = Field(..., min_length=1)
    dataset_id: str = Field(..., min_length=1)
    dataset_fingerprint: str = Field(..., min_length=1)
    provenance: ProvenanceFingerprints
    verdict: VerdictMetrics
    retrieval: RetrievalMetrics | None = Field(default=None)
    citation: CitationMetrics | None = Field(default=None)
    latency: LatencyMetrics
    example_results: list[ExampleResult] = Field(..., min_length=1)
    example_count: int = Field(..., gt=0)


class AblatedComponent(str, Enum):
    """A pipeline component whose contribution the ablation matrix isolates."""

    QUESTION_GENERATION = "QUESTION_GENERATION"
    BM25 = "BM25"
    RRF = "RRF"


class ContributionDelta(BaseModel):
    """The measured contribution of one ablated component on one metric."""

    model_config = ConfigDict(frozen=True)

    component: AblatedComponent
    metric_name: str = Field(..., min_length=1)
    with_value: float
    without_value: float
    delta: float


class AblationReport(BaseModel):
    """The full set of per-variant reports plus the derived contribution deltas."""

    model_config = ConfigDict(frozen=True)

    reports: list[EvaluationReport] = Field(..., min_length=1)
    deltas: list[ContributionDelta] = Field(default_factory=list)
