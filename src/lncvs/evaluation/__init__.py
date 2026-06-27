"""Evaluation Framework: ledger-driven, read-only scoring of pipeline runs against gold datasets.

evaluation/ is the most-downstream module in the dependency chain -- it may
import from every other lncvs module and nothing imports from it. The
PipelineRunner here is evaluation infrastructure only, not production
orchestration; LangGraph (a later phase) will replace or absorb it without
changing any metric function or schema in this package.
"""

from lncvs.evaluation.config import AblationVariant, EvaluationConfig, FusionStrategy, standard_ablation_matrix
from lncvs.evaluation.dataset import load_dataset, map_spans_to_chunks
from lncvs.evaluation.fingerprint import ledger_fingerprint
from lncvs.evaluation.fusion_baselines import round_robin_fuse
from lncvs.evaluation.runner import PipelineRunner
from lncvs.evaluation.service import EvaluationHarness

__all__ = [
    "AblationVariant",
    "EvaluationConfig",
    "EvaluationHarness",
    "FusionStrategy",
    "PipelineRunner",
    "ledger_fingerprint",
    "load_dataset",
    "map_spans_to_chunks",
    "round_robin_fuse",
    "standard_ablation_matrix",
]
