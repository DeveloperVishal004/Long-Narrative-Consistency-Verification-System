"""PipelineRunner: thin, ablation-parameterized execution of the Phase 0-5 pipeline.

This is evaluation infrastructure, not production orchestration -- it
exists solely so EvaluationHarness can produce a scored EvidenceLedger for
each (example, AblationVariant) pair. LangGraph (a later phase) will
replace or absorb this runner; evaluation/metrics and EvaluationHarness
consume only the EvidenceLedger it returns, so neither changes when that
happens.

Models (embedder, NLI model, LLM clients) and their caches are injected
once at construction and shared across every run() call, so repeated
evaluation/ablation runs do not pay for redundant inference. Each run()
call builds a fresh ChromaIndex/BM25Index scoped to a unique collection
name -- index construction itself is not cached, since narratives differ
across examples; re-indexing the same narrative across ablation variants is
a known, accepted cost in the MVP (see Phase 6 architecture review Risk 5).

Collection names use uuid4, not a content hash or per-instance counter.
ChromaDB's EphemeralClient persists collections by name across separate
client instances within the same process -- a per-instance counter
restarting at 1 collides the moment a second PipelineRunner is constructed
(as two EvaluationHarness runs, or two examples in the same dataset, both
would), silently leaking one narrative's chunks into another's retrieval
results. Collection names carry no audit/determinism requirement (they are
not stored in the ledger, unlike chunk_id/evidence_id), so uuid4 is the
correct tool here, not a violation of the content-hash-ID discipline.
"""

import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

from lncvs.chunking import ChunkingConfig, chunk_document
from lncvs.fusion import FusionConfig, fuse_evidence
from lncvs.orchestration.fusion_baselines import round_robin_fuse
from lncvs.indexing import BM25Index, ChromaIndex, Embedder
from lncvs.ingestion import load_and_clean_narrative
from lncvs.ledger import LedgerService
from lncvs.llm import LLMClient
from lncvs.reasoning.decomposition import DecompositionConfig, LLMClaimDecomposer, make_source_claim_id
from lncvs.reasoning.nli import CrossEncoderNLIVerifier, NLIModel
from lncvs.reasoning.questions import LLMQuestionGenerator, QuestionGenerationConfig
from lncvs.retrieval import (
    BM25Retriever,
    RetrievalConfig,
    RetrievalOrchestrator,
    SemanticRetriever,
    build_retrieval_queries,
)
from lncvs.rules import RuleEngineConfig, ThresholdRuleEngine, classify
from lncvs.schemas import (
    AblationVariant,
    AtomicClaim,
    EvidenceLedger,
    FusedEvidence,
    FusionStrategy,
    NLIResult,
    ProbeQuestion,
)


@runtime_checkable
class LedgerProducer(Protocol):
    """Contract satisfied by both PipelineRunner and orchestration.LangGraphPipeline.

    EvaluationHarness depends only on this protocol, never on a concrete
    runner -- this is what lets Phase 7 swap PipelineRunner for the
    LangGraph-backed runner without touching any evaluation/ logic, only
    the type annotation on EvaluationHarness.__init__.
    """

    @property
    def chunking_config(self) -> ChunkingConfig: ...

    def run(self, narrative_path: Path, original_claim: str, variant: AblationVariant) -> EvidenceLedger: ...


class PipelineRunner:
    """Runs the full Phase 0-5 pipeline once per (narrative, claim, variant), returning a ledger."""

    def __init__(
        self,
        embedder: Embedder,
        nli_model: NLIModel,
        decomposition_llm: LLMClient,
        question_llm: LLMClient,
        decomposition_config: DecompositionConfig,
        question_config: QuestionGenerationConfig,
        rule_config: RuleEngineConfig,
        chunking_config: ChunkingConfig,
        retrieval_top_k: int = 10,
        fusion_config: FusionConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._nli_model = nli_model
        self._decomposition_llm = decomposition_llm
        self._question_llm = question_llm
        self._decomposition_config = decomposition_config
        self._question_config = question_config
        self._rule_config = rule_config
        self._chunking_config = chunking_config
        self._retrieval_top_k = retrieval_top_k
        self._fusion_config = fusion_config or FusionConfig()

    @property
    def chunking_config(self) -> ChunkingConfig:
        return self._chunking_config

    def run(self, narrative_path: Path, original_claim: str, variant: AblationVariant) -> EvidenceLedger:
        collection_suffix = f"eval-{uuid.uuid4().hex}"

        document = load_and_clean_narrative(narrative_path, source_id=str(narrative_path))
        chunks = chunk_document(document, self._chunking_config)

        chroma_index = ChromaIndex(embedder=self._embedder, collection_name=f"semantic-{collection_suffix}")
        chroma_index.index(chunks)
        retrievers = [SemanticRetriever(chroma_index)]

        if variant.use_bm25:
            bm25_index = BM25Index(collection_name=f"lexical-{collection_suffix}")
            bm25_index.index(chunks)
            retrievers.append(BM25Retriever(bm25_index))

        ledger = EvidenceLedger(original_claim=original_claim)
        service = LedgerService(ledger)

        decomposer = LLMClaimDecomposer(self._decomposition_llm, self._decomposition_config)
        parent_id = make_source_claim_id(original_claim)
        atomic_claims = decomposer.decompose(original_claim)
        service.record_atomic_claims(parent_id, atomic_claims)

        all_questions: list[ProbeQuestion] = []
        if variant.use_question_generation:
            generator = LLMQuestionGenerator(self._question_llm, self._question_config)
            for claim in atomic_claims:
                all_questions.extend(generator.generate(claim))
        service.record_probe_questions(all_questions)

        queries = build_retrieval_queries(atomic_claims, all_questions)
        service.record_retrieval_queries(queries)

        orchestrator = RetrievalOrchestrator(retrievers, RetrievalConfig(top_k=self._retrieval_top_k))
        evidence = orchestrator.retrieve_for_queries(queries)
        service.record_retrieved_evidence(evidence)

        if variant.fusion_strategy is FusionStrategy.ROUND_ROBIN:
            fused = round_robin_fuse(service.ledger.retrieved_evidence, self._fusion_config.top_k_fused)
        else:
            fused = fuse_evidence(service.ledger.retrieved_evidence, self._fusion_config)
        service.record_fused_evidence(fused)

        nli_results = self._verify(atomic_claims, service.ledger.fused_evidence)
        service.record_nli_results(nli_results)

        claim_ids = [claim.claim_id for claim in atomic_claims]
        outcome = classify(service.ledger.nli_results, claim_ids, self._rule_config)
        service.record_classification(outcome.contradictions, outcome.supporting_evidence, outcome.unsupported_claim_ids)

        engine = ThresholdRuleEngine(self._rule_config)
        verdict = engine.evaluate(service.ledger)
        service.set_final_verdict(verdict)

        return service.ledger

    def _verify(self, atomic_claims: list[AtomicClaim], fused_evidence: list[FusedEvidence]) -> list[NLIResult]:
        verifier = CrossEncoderNLIVerifier(self._nli_model)
        fused_by_claim: dict[str, list[FusedEvidence]] = {}
        for record in fused_evidence:
            fused_by_claim.setdefault(record.atomic_claim_id, []).append(record)

        results: list[NLIResult] = []
        for claim in atomic_claims:
            results.extend(verifier.verify(claim, fused_by_claim.get(claim.claim_id, [])))
        return results
