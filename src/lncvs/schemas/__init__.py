"""All shared Pydantic models and enums for LNCVS.

This is the universal leaf module: it depends on nothing else in lncvs, and
every other module may depend on it. No module outside schemas/ may define
a competing type for any of the names exported here.
"""

from lncvs.schemas.chunk import DocumentChunk
from lncvs.schemas.claims import AtomicClaim, ProbeQuestion
from lncvs.schemas.document import RawDocument
from lncvs.schemas.enums import FactVerificationLabel, FusionStrategy, NLILabel, PipelineStage, QueryOrigin, RetrievalSource, VerdictEnum
from lncvs.schemas.evaluation import (
    AblatedComponent,
    AblationReport,
    AblationVariant,
    CitationMetrics,
    ConfusionCell,
    ConfusionMatrix,
    ContributionDelta,
    EvaluationDataset,
    EvaluationReport,
    ExampleResult,
    GoldExample,
    GoldSpan,
    LatencyMetrics,
    PerClassMetric,
    ProvenanceFingerprints,
    RankCutoffMetric,
    RetrievalMetrics,
    StageLatency,
    VerdictMetrics,
    standard_ablation_matrix,
)
from lncvs.schemas.evidence import Contradiction, FusedEvidence, RetrievedEvidence, SupportingEvidence
from lncvs.schemas.fact_verification import FactVerification
from lncvs.schemas.graph import (
    EntityRecord,
    EntityRelation,
    EntityType,
    EventParticipation,
    EventRecord,
    ParticipantRole,
    RelationType,
    TemporalKind,
)
from lncvs.schemas.ledger import EvidenceLedger
from lncvs.schemas.nli import NLIResult
from lncvs.schemas.provenance import Provenance
from lncvs.schemas.reasoning import LedgerEvent, ReasoningStep
from lncvs.schemas.retrieval_query import RetrievalQuery
from lncvs.schemas.state import ControlState, GraphState, StageError
from lncvs.schemas.verdict import FinalVerdict

__all__ = [
    "AblatedComponent",
    "AblationReport",
    "AblationVariant",
    "AtomicClaim",
    "CitationMetrics",
    "ConfusionCell",
    "ConfusionMatrix",
    "Contradiction",
    "ContributionDelta",
    "ControlState",
    "DocumentChunk",
    "EvaluationDataset",
    "EvaluationReport",
    "EntityRecord",
    "EntityRelation",
    "EntityType",
    "EventParticipation",
    "EventRecord",
    "EvidenceLedger",
    "ExampleResult",
    "FactVerification",
    "FactVerificationLabel",
    "FinalVerdict",
    "FusedEvidence",
    "FusionStrategy",
    "GoldExample",
    "GoldSpan",
    "GraphState",
    "LatencyMetrics",
    "LedgerEvent",
    "NLILabel",
    "NLIResult",
    "ParticipantRole",
    "PerClassMetric",
    "PipelineStage",
    "ProbeQuestion",
    "Provenance",
    "ProvenanceFingerprints",
    "QueryOrigin",
    "RankCutoffMetric",
    "RawDocument",
    "ReasoningStep",
    "RelationType",
    "RetrievalMetrics",
    "RetrievalQuery",
    "RetrievalSource",
    "RetrievedEvidence",
    "StageError",
    "StageLatency",
    "SupportingEvidence",
    "TemporalKind",
    "VerdictEnum",
    "VerdictMetrics",
    "standard_ablation_matrix",
]
