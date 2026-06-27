"""LangGraph node callables: each node calls the identical underlying service
functions lncvs.evaluation.runner.PipelineRunner.run() calls, in the same
order, mutating the ledger only through LedgerService. This identity is what
makes "preserve behavior" a structural guarantee rather than a hope -- see
tests/orchestration/test_graph_equivalence.py.

ingest_and_index deliberately writes NOTHING to the ledger (matches the
oracle: PipelineRunner builds its indices as local variables before any
LedgerService call). Every other node's only side effect is exactly one
LedgerService.record_*() call (or two, for retrieve/verdict), exactly
mirroring PipelineRunner's call sequence.

Per-backend retrieval failure tolerance (degraded_sources) is NOT wired
here: RetrievalOrchestrator does not catch per-backend exceptions today (no
production behavior to preserve), and adding that tolerance would be a new
capability, which Phase 7 is explicitly scoped to avoid. A failing
retriever propagates like any other node exception, into the generic error
boundary below.
"""

import logging
import uuid
from datetime import datetime, timezone
from functools import wraps
from typing import Callable

from langchain_core.runnables import RunnableConfig

from lncvs.indexing import BM25Index, ChromaIndex
from lncvs.ingestion import load_and_clean_narrative
from lncvs.ledger import LedgerService
from lncvs.orchestration.fusion_baselines import round_robin_fuse
from lncvs.orchestration.resources import PipelineResources, RunContext
from lncvs.orchestration.state_channels import GraphChannels
from lncvs.chunking import chunk_document
from lncvs.fusion import fuse_evidence
from lncvs.reasoning.decomposition import LLMClaimDecomposer, make_source_claim_id
from lncvs.reasoning.nli import CrossEncoderNLIVerifier
from lncvs.reasoning.questions import LLMQuestionGenerator
from lncvs.retrieval import BM25Retriever, RetrievalConfig, RetrievalOrchestrator, SemanticRetriever, build_retrieval_queries
from lncvs.rules import ThresholdRuleEngine, classify
from lncvs.schemas import FusedEvidence, FusionStrategy, PipelineStage, ProbeQuestion, StageError

logger = logging.getLogger(__name__)


def _advance(control, stage: PipelineStage):
    return control.model_copy(update={"current_stage": stage})


def _resources(config: RunnableConfig) -> PipelineResources:
    return config["configurable"]["resources"]


def _run_context(config: RunnableConfig) -> RunContext:
    return config["configurable"]["run_context"]


def _node_error_boundary(stage: PipelineStage) -> Callable:
    """Wrap a node so any exception becomes a routed, recorded StageError --
    never a crashed run, never a silently swallowed failure. Single place
    that defines how node errors become orchestration-visible state."""

    def decorator(func: Callable[[GraphChannels, RunnableConfig], dict]) -> Callable:
        @wraps(func)
        def wrapper(state: GraphChannels, config: RunnableConfig) -> dict:
            try:
                return func(state, config)
            except Exception as exc:
                logger.error("Node %s failed at stage %s: %s", func.__name__, stage.value, exc)
                new_control = state.control.model_copy(
                    update={
                        "current_stage": PipelineStage.ERROR,
                        "errors": [
                            *state.control.errors,
                            StageError(stage=stage, message=str(exc), timestamp=datetime.now(timezone.utc)),
                        ],
                    }
                )
                return {"control": new_control}

        return wrapper

    return decorator


@_node_error_boundary(PipelineStage.INGESTION)
def ingest_and_index(state: GraphChannels, config: RunnableConfig) -> dict:
    """Ingest, chunk, and index the narrative. Writes nothing to the ledger --
    matches PipelineRunner, which builds its indices as local variables
    before any LedgerService call."""
    resources = _resources(config)
    run_context = _run_context(config)
    narrative_path = config["configurable"]["narrative_path"]

    document = load_and_clean_narrative(narrative_path, source_id=str(narrative_path))
    chunks = chunk_document(document, resources.chunking_config)

    collection_suffix = f"eval-{uuid.uuid4().hex}"
    chroma_index = ChromaIndex(embedder=resources.embedder, collection_name=f"semantic-{collection_suffix}")
    chroma_index.index(chunks)
    run_context.semantic_retriever = SemanticRetriever(chroma_index)

    if run_context.variant.use_bm25:
        bm25_index = BM25Index(collection_name=f"lexical-{collection_suffix}")
        bm25_index.index(chunks)
        run_context.lexical_retriever = BM25Retriever(bm25_index)

    return {"control": _advance(state.control, PipelineStage.CLAIM_DECOMPOSITION)}


@_node_error_boundary(PipelineStage.CLAIM_DECOMPOSITION)
def decompose_claim(state: GraphChannels, config: RunnableConfig) -> dict:
    resources = _resources(config)
    service = LedgerService(state.ledger)

    decomposer = LLMClaimDecomposer(resources.decomposition_llm, resources.decomposition_config)
    parent_id = make_source_claim_id(state.ledger.original_claim)
    atomic_claims = decomposer.decompose(state.ledger.original_claim)
    service.record_atomic_claims(parent_id, atomic_claims)

    return {"ledger": service.ledger, "control": _advance(state.control, PipelineStage.QUESTION_GENERATION)}


@_node_error_boundary(PipelineStage.QUESTION_GENERATION)
def generate_questions(state: GraphChannels, config: RunnableConfig) -> dict:
    resources = _resources(config)
    run_context = _run_context(config)
    service = LedgerService(state.ledger)

    all_questions: list[ProbeQuestion] = []
    if run_context.variant.use_question_generation:
        generator = LLMQuestionGenerator(resources.question_llm, resources.question_config)
        for claim in state.ledger.atomic_claims:
            all_questions.extend(generator.generate(claim))
    service.record_probe_questions(all_questions)

    return {"ledger": service.ledger, "control": _advance(state.control, PipelineStage.RETRIEVAL)}


@_node_error_boundary(PipelineStage.RETRIEVAL)
def retrieve(state: GraphChannels, config: RunnableConfig) -> dict:
    resources = _resources(config)
    run_context = _run_context(config)
    service = LedgerService(state.ledger)

    retrievers = [run_context.semantic_retriever]
    if run_context.variant.use_bm25:
        retrievers.append(run_context.lexical_retriever)

    queries = build_retrieval_queries(state.ledger.atomic_claims, state.ledger.probe_questions)
    service.record_retrieval_queries(queries)

    orchestrator = RetrievalOrchestrator(retrievers, RetrievalConfig(top_k=resources.retrieval_top_k))
    evidence = orchestrator.retrieve_for_queries(queries)
    service.record_retrieved_evidence(evidence)

    return {"ledger": service.ledger, "control": _advance(state.control, PipelineStage.FUSION)}


@_node_error_boundary(PipelineStage.FUSION)
def fuse(state: GraphChannels, config: RunnableConfig) -> dict:
    resources = _resources(config)
    run_context = _run_context(config)
    service = LedgerService(state.ledger)

    if run_context.variant.fusion_strategy is FusionStrategy.ROUND_ROBIN:
        fused = round_robin_fuse(state.ledger.retrieved_evidence, resources.fusion_config_or_default().top_k_fused)
    else:
        fused = fuse_evidence(state.ledger.retrieved_evidence, resources.fusion_config_or_default())
    service.record_fused_evidence(fused)

    return {"ledger": service.ledger, "control": _advance(state.control, PipelineStage.NLI_VERIFICATION)}


@_node_error_boundary(PipelineStage.NLI_VERIFICATION)
def verify_nli(state: GraphChannels, config: RunnableConfig) -> dict:
    resources = _resources(config)
    service = LedgerService(state.ledger)

    verifier = CrossEncoderNLIVerifier(resources.nli_model)
    fused_by_claim: dict[str, list[FusedEvidence]] = {}
    for record in state.ledger.fused_evidence:
        fused_by_claim.setdefault(record.atomic_claim_id, []).append(record)

    nli_results = []
    for claim in state.ledger.atomic_claims:
        nli_results.extend(verifier.verify(claim, fused_by_claim.get(claim.claim_id, [])))
    service.record_nli_results(nli_results)

    return {"ledger": service.ledger, "control": _advance(state.control, PipelineStage.RULE_ENGINE)}


@_node_error_boundary(PipelineStage.RULE_ENGINE)
def verdict(state: GraphChannels, config: RunnableConfig) -> dict:
    resources = _resources(config)
    service = LedgerService(state.ledger)

    claim_ids = [claim.claim_id for claim in state.ledger.atomic_claims]
    outcome = classify(state.ledger.nli_results, claim_ids, resources.rule_config)
    service.record_classification(outcome.contradictions, outcome.supporting_evidence, outcome.unsupported_claim_ids)

    engine = ThresholdRuleEngine(resources.rule_config)
    final_verdict = engine.evaluate(service.ledger)
    service.set_final_verdict(final_verdict)

    return {"ledger": service.ledger, "control": _advance(state.control, PipelineStage.COMPLETE)}


def error_sink(state: GraphChannels, config: RunnableConfig) -> dict:
    """Terminal node for the error path. A no-op: the failing node already
    recorded the StageError and set control.current_stage to ERROR. This
    node exists only to give the error path a named destination distinct
    from the happy-path END, so a failed run returns the partial ledger
    with no fabricated verdict."""
    return {}


def route_after(next_node: str) -> Callable[[GraphChannels], str]:
    """Conditional-edge router shared by every node: route to error_sink if
    the upstream node set control.current_stage to ERROR, else continue."""

    def _route(state: GraphChannels) -> str:
        return "error_sink" if state.control.current_stage is PipelineStage.ERROR else next_node

    return _route
