"""Evaluation harness configuration.

AblationVariant, FusionStrategy, and standard_ablation_matrix moved to
schemas/ in Phase 7 (orchestration/ needs them too, and must not import
from evaluation/) -- re-exported here so every existing
`from lncvs.evaluation import AblationVariant, ...` import keeps working
unchanged. EvaluationConfig itself stays here: it is genuinely
evaluation-only (nothing outside evaluation/ reads it).
"""

import hashlib

from pydantic import BaseModel, ConfigDict, Field

from lncvs.schemas import AblationVariant, FusionStrategy, standard_ablation_matrix

__all__ = ["AblationVariant", "EvaluationConfig", "FusionStrategy", "standard_ablation_matrix"]


class EvaluationConfig(BaseModel):
    """Configures a single EvaluationHarness run, independent of any one AblationVariant.

    Deliberately holds only what EvaluationHarness itself reads. Chunking,
    retrieval top_k, and rule-engine thresholds are NOT duplicated here --
    they are owned by ChunkingConfig/RetrievalConfig/RuleEngineConfig,
    injected directly into PipelineRunner by its caller. Duplicating them
    here would create two sources of truth for the same setting.
    """

    model_config = ConfigDict(frozen=True)

    k_cutoffs: list[int] = Field(default_factory=lambda: [5, 10], description="Rank cutoffs for Recall@k/Precision@k.")
    seed: int = Field(default=0, description="Recorded for provenance; the pipeline itself is deterministic by construction.")
    output_dir: str = Field(
        default="evaluation_runs",
        description="Directory persisted reports and ledgers are written to. Only used when persist_ledgers is True.",
    )
    persist_ledgers: bool = Field(
        default=False,
        description=(
            "Single gate for ALL filesystem writes EvaluationHarness performs. False (default): "
            "evaluate_variant() writes nothing to disk and ExampleResult.ledger_path stays None. "
            "True: each example's EvidenceLedger is written under output_dir/ledgers/, and the "
            "finished EvaluationReport is written under output_dir/, both via lncvs.evaluation.reporting."
        ),
    )

    def fingerprint(self) -> str:
        """Deterministic identifier for this exact configuration."""
        digest_input = f"{self.k_cutoffs}:{self.seed}".encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()[:16]
