"""Closed vocabularies shared across the LNCVS pipeline.

Per CLAUDE.md, every closed vocabulary in this system is an enum, never a
bare string. RetrievalSource.GRAPH (added in Phase 8 / Version 2) is the
one exception to "no graph references in Version 1 code": its addition was
an explicit, recorded project-owner decision (the V2 entry gate was
declared satisfied and graph work was authorized to land in this same
repository, not a separate one) — see CLAUDE.md's Version 2 Roadmap section
and the Phase 8 architecture review. The graph retriever that produces this
source is opt-in: no V1 code path constructs or wires it by default.
"""

from enum import Enum


class VerdictEnum(str, Enum):
    """Final verdict produced by the deterministic rule engine."""

    CONSISTENT = "CONSISTENT"
    CONTRADICTORY = "CONTRADICTORY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class NLILabel(str, Enum):
    """Label produced by NLI verification for a single (premise, hypothesis) pair."""

    ENTAILMENT = "ENTAILMENT"
    CONTRADICTION = "CONTRADICTION"
    NEUTRAL = "NEUTRAL"


class FactVerificationLabel(str, Enum):
    """Label produced by a FactVerifier for a single (atomic fact, evidence) pair
    (Phase H2). A separate vocabulary from NLILabel, not a renaming of it:
    FactVerifier implementations (CrossEncoderFactVerifier today,
    LLMFactVerifier in Phase H3) reason in terms of fact-verification
    semantics ("is this fact mentioned at all?"), which maps onto, but is
    conceptually prior to, NLI's entailment/contradiction/neutral vocabulary.
    The mapping is fixed and one-directional (fact_verification/compat.py):
    SUPPORTED -> ENTAILMENT, CONTRADICTED -> CONTRADICTION,
    NOT_MENTIONED -> NEUTRAL.
    """

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_MENTIONED = "NOT_MENTIONED"


class RetrievalSource(str, Enum):
    """Backend that produced a piece of retrieved evidence.

    SEMANTIC (ChromaDB) and LEXICAL (BM25) are Version 1. GRAPH (Version 2,
    Phase 8) is produced only by lncvs.graph.GraphRetriever, an index over
    the same chunk-ID space — never a new evidence store. See module
    docstring for the gate-decision record.
    """

    SEMANTIC = "SEMANTIC"
    LEXICAL = "LEXICAL"
    GRAPH = "GRAPH"


class QueryOrigin(str, Enum):
    """Whether a RetrievalQuery's text came from an atomic claim itself or a probe question.

    CLAIM queries always exist (one per atomic claim) — this is why an
    empty probe-question set is non-fatal: the claim text is always a
    fallback query. QUESTION queries are additive coverage.
    """

    CLAIM = "CLAIM"
    QUESTION = "QUESTION"


class FusionStrategy(str, Enum):
    """Strategy for combining per-source retrieval results into ranked evidence.

    RRF (Reciprocal Rank Fusion) is the only production strategy — see
    lncvs.fusion.rrf. ROUND_ROBIN is an evaluation-only baseline (see
    lncvs.evaluation.fusion_baselines), used solely to isolate RRF's
    measured contribution in the Phase 6 ablation matrix; lncvs.fusion
    never implements it. Lives here (not in evaluation/) because Phase 7's
    orchestration/ also needs it to decide a graph node's fusion path,
    and orchestration/ must not import from evaluation/.
    """

    RRF = "RRF"
    ROUND_ROBIN = "ROUND_ROBIN"


class PipelineStage(str, Enum):
    """Named stages of the LNCVS verification pipeline.

    Used by ControlState for orchestration tracking and by LedgerEvent /
    ReasoningStep for the audit trail. ERROR is a terminal stage reached via
    failure routing, not a step in the normal sequence.
    """

    INGESTION = "INGESTION"
    CHUNKING = "CHUNKING"
    INDEXING = "INDEXING"
    CLAIM_DECOMPOSITION = "CLAIM_DECOMPOSITION"
    QUESTION_GENERATION = "QUESTION_GENERATION"
    RETRIEVAL = "RETRIEVAL"
    FUSION = "FUSION"
    NLI_VERIFICATION = "NLI_VERIFICATION"
    RULE_ENGINE = "RULE_ENGINE"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
