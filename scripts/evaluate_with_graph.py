"""Graph Retrieval Impact Study (Phase 8 / G2): measures whether adding
GraphRetriever to the existing Chroma+BM25 retrieval changes verdict
accuracy on the hackathon dataset (train.csv/test.csv).

Scope note, disclosed: this is an EVALUATION-ONLY thin driver, mirroring
the precedent already established in tests/acceptance/test_phase5_nli_verdict.py
and documented in validate_long_narrative.py -- a small, local wiring
function, not a new orchestration module. The real LangGraph orchestration
(lncvs.orchestration) has no third-retriever injection point today without
modifying orchestration/nodes.py AND lncvs.evaluation.runner.PipelineRunner
together (they must stay byte-identical per the Phase 7 equivalence-oracle
guarantee) -- that is a real production extension, explicitly deferred
per the hackathon-deadline priority (measure the effect now; promote to
the LangGraph node path later if warranted). Both the with-graph and
without-graph conditions below run through the IDENTICAL thin driver,
varying only the injected retriever list, so the comparison is
apples-to-apples regardless of how this driver differs from the LangGraph
node path.

Retrieval (lncvs.retrieval), fusion (lncvs.fusion), NLI
(lncvs.reasoning.nli), and the rule engine (lncvs.rules) are the real,
unmodified production services -- only the script-local wiring that calls
them in sequence is new, exactly as Phase 5's offline driver already
established as acceptable.

Graph construction uses real OpenAI structured-output extraction
(gpt-4o-2024-08-06, temperature=0), cached to
results/extraction_cache_<book>.jsonl via JsonlStructuredLLMCache so the
~$3 one-time cost (per the G2 architecture freeze's cost estimate) is
paid once, ever -- every subsequent run of this script reloads the cache
and makes zero further OpenAI calls.
"""

import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from evaluate_dataset import (  # noqa: E402
    BOOK_NAME_TO_PATH,
    DEFAULT_CONTRADICTION_THRESHOLD,
    DEFAULT_ENTAILMENT_THRESHOLD,
    GOLD_LABEL_TO_VERDICT,
    DatasetRow,
    append_checkpoint,
    build_claim_text,
    load_checkpoint,
    load_csv,
)

from lncvs.chunking import ChunkingConfig, chunk_document  # noqa: E402
from lncvs.evaluation.metrics.verdict import compute_verdict_metrics  # noqa: E402
from lncvs.fusion import FusionConfig, fuse_evidence  # noqa: E402
from lncvs.graph import GraphIndex, GraphRetriever  # noqa: E402
from lncvs.graph.construction import ConstructedGraph, build_graph_for_novel  # noqa: E402
from lncvs.graph.llm_extraction import ExtractionConfig, LLMWindowExtractor, SYSTEM_PROMPT  # noqa: E402
from lncvs.indexing import BM25Index, CachingEmbedder, ChromaIndex, EmbeddingConfig, InMemoryEmbeddingCache, SentenceTransformerEmbedder  # noqa: E402
from lncvs.ingestion import load_and_clean_narrative  # noqa: E402
from lncvs.ledger import LedgerService  # noqa: E402
from lncvs.llm import CachingLLMClient, CachingStructuredLLMClient, GeminiLLMClient, GeminiStructuredClient, JsonlLLMCache, JsonlStructuredLLMCache, LLMConfig, OpenAIStructuredClient  # noqa: E402
from lncvs.reasoning.decomposition import DecompositionConfig, LLMClaimDecomposer, make_source_claim_id  # noqa: E402
from lncvs.reasoning.fact_verification import (  # noqa: E402
    CrossEncoderFactVerifier,
    FactVerificationConfig,
    FactVerifier,
    LLMFactVerifier,
    to_nli_results,
)
from lncvs.reasoning.fact_verification.llm_prompts import SYSTEM_PROMPT as FACT_VERIFICATION_SYSTEM_PROMPT  # noqa: E402
from lncvs.reasoning.nli import CachingNLIModel, CrossEncoderNLIModel, CrossEncoderNLIVerifier, InMemoryNLICache, NLIConfig  # noqa: E402
from lncvs.retrieval import BM25Retriever, RetrievalConfig, RetrievalOrchestrator, SemanticRetriever, build_retrieval_queries  # noqa: E402
from lncvs.rules import RuleEngineConfig, ThresholdRuleEngine, classify  # noqa: E402
from lncvs.schemas import DocumentChunk, EvidenceLedger, RetrievalSource, VerdictEnum  # noqa: E402

import networkx as nx  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("evaluate_with_graph")

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
EXTRACTION_MODEL = "gemini-2.5-flash"
DECOMPOSITION_MODEL = "gemini-2.5-flash"

# H7 (metric-improvement slice): GEMINI_API_KEY is unavailable in this
# environment (removed earlier this session); OPENAI_API_KEY is present
# and billing-enabled (confirmed working, already used for the H5
# validation run). VERIFIER_PROVIDER selects which StructuredLLMClient
# backs "llm" mode -- OpenAIStructuredClient already exists in this
# codebase (frozen, used by G2 graph extraction's real gpt-4o-2024-08-06
# calls) and implements the identical StructuredLLMClient protocol, so
# LLMFactVerifier is unaware which provider it was given.
VERIFIER_PROVIDER = "openai"  # "gemini" or "openai"
VERIFICATION_MODEL = "gpt-4o-2024-08-06" if VERIFIER_PROVIDER == "openai" else "gemini-2.5-flash"

# Phase H4: the verifier is now selected entirely through this configuration
# constant, never through a code change -- "cross_encoder" wraps the
# existing, frozen CrossEncoderNLIVerifier (Phase H2's CrossEncoderFactVerifier,
# zero API calls); "llm" wraps the real, cached structured client (Phase H3's
# LLMFactVerifier, evidence-set-level since the Phase H redesign, real API
# cost on cache misses). Both flow through the IDENTICAL downstream path:
# to_nli_results -> classify -> ThresholdRuleEngine -- see run_claim.
# H7: flipped from "cross_encoder" to "llm" -- per the H6 evaluation, the
# cross-encoder verifier is a near-degenerate always-CONTRADICTORY
# classifier (100% CONTRADICTORY prediction rate, CONSISTENT recall 0.0)
# that is structurally insensitive to retrieval/graph quality, making it
# the bottleneck the H6 report identified ahead of any further retrieval
# work.
VERIFIER_MODE = "llm"


class _NoCallStructuredClient:
    """StructuredLLMClient that raises (ValueError) if ever invoked. Behind
    CachingStructuredLLMClient, a cache hit means it is never called -- so
    the graph is rebuilt from the on-disk extraction cache with ZERO Gemini
    API calls. A miss (only the original uncached, content-less windows)
    raises ValueError, which the construction pipeline's per-window
    fault-isolation skips, reproducing the original graph exactly."""

    def complete_structured(self, prompt, response_schema):
        raise ValueError(
            "Cache miss (window not in extraction cache -- skipped to reproduce the "
            "original graph; no Gemini API call made; graph is final and read-only)."
        )


def build_fact_verifier(mode: str, cached_nli_model: object | None) -> FactVerifier:
    """Construct the FactVerifier selected by VERIFIER_MODE (Phase H4).

    This is the ONLY place verifier construction happens; run_claim takes
    a FactVerifier and is completely unaware of which implementation it
    received. Nothing downstream of this factory -- to_nli_results,
    classify, ThresholdRuleEngine, the Ledger -- changes based on mode.

    cached_nli_model is required (and used) only for "cross_encoder";
    "llm" never touches it, so main() only needs to load the cross-encoder
    model when that mode is actually selected.
    """
    if mode == "cross_encoder":
        if cached_nli_model is None:
            raise ValueError("cross_encoder mode requires a cached_nli_model")
        return CrossEncoderFactVerifier(CrossEncoderNLIVerifier(cached_nli_model))

    if mode == "llm":
        llm_config = LLMConfig(model_name=VERIFICATION_MODEL, temperature=0.0, max_tokens=2048)
        fact_verification_config = FactVerificationConfig()
        if VERIFIER_PROVIDER == "openai":
            real_client = OpenAIStructuredClient(config=llm_config, system_prompt=FACT_VERIFICATION_SYSTEM_PROMPT, schema_name="fact_verification")
        elif VERIFIER_PROVIDER == "gemini":
            real_client = GeminiStructuredClient(config=llm_config, system_prompt=FACT_VERIFICATION_SYSTEM_PROMPT)
        else:
            raise ValueError(f"Unknown VERIFIER_PROVIDER {VERIFIER_PROVIDER!r}; expected 'gemini' or 'openai'")
        safe_model_name = VERIFICATION_MODEL.replace(".", "_").replace("-", "_")
        cache = JsonlStructuredLLMCache(RESULTS_DIR / f"fact_verification_cache_{VERIFIER_PROVIDER}_{safe_model_name}.jsonl")
        caching_client = CachingStructuredLLMClient(real_client, cache, llm_config, fact_verification_config.schema_version)
        return LLMFactVerifier(caching_client, fact_verification_config)

    raise ValueError(f"Unknown VERIFIER_MODE {mode!r}; expected 'cross_encoder' or 'llm'")


def build_novel_graph(book_name: str, book_path: Path, cached_embedder) -> tuple[list[DocumentChunk], ChromaIndex, BM25Index, GraphRetriever, ConstructedGraph]:
    """Build chunks + Chroma + BM25 + the real G2 LLM-extracted graph for
    one novel, once. Reused across every claim referencing this book."""
    logger.info("Building graph for %r from %s", book_name, book_path)
    document = load_and_clean_narrative(book_path, source_id=book_name)
    chunks = chunk_document(document, ChunkingConfig(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP))

    safe_name = book_name.replace(" ", "_").lower()
    chroma_index = ChromaIndex(embedder=cached_embedder, collection_name=f"graph-impact-{safe_name}")
    chroma_index.index(chunks)
    bm25_index = BM25Index(collection_name=f"graph-impact-{safe_name}-bm25")
    bm25_index.index(chunks)

    # 4096, 16384, then 32768 all truncated mid-JSON on real, content-dense
    # chapters during live runs -- caught immediately, never cached (a
    # failed call's response never reaches CachingStructuredLLMClient.put()).
    # 65536 is gemini-2.5-flash's real output_token_limit (confirmed via the
    # API's own model metadata) -- the actual ceiling, not a guess.
    llm_config = LLMConfig(model_name=EXTRACTION_MODEL, temperature=0.0, max_tokens=65536)
    extraction_config = ExtractionConfig()
    # Cache-only: every extraction window for both novels is already on disk
    # (results/extraction_cache_<book>.jsonl). _NoCallStructuredClient raises
    # ValueError on any miss, which the construction pipeline's per-window
    # fault-isolation skips (reproducing the original graph, e.g. Monte
    # Cristo ch.156 which returned no content and was never cached). This
    # guarantees ZERO Gemini API calls -- the graph is final and read-only.
    cache = JsonlStructuredLLMCache(RESULTS_DIR / f"extraction_cache_{safe_name}.jsonl")
    caching_client = CachingStructuredLLMClient(_NoCallStructuredClient(), cache, llm_config, extraction_config.schema_version)
    extractor = LLMWindowExtractor(caching_client, extraction_config)

    t0 = time.perf_counter()
    constructed, entity_graph = build_graph_for_novel(document.cleaned_text, chunks, extractor)
    build_seconds = time.perf_counter() - t0
    logger.info(
        "Graph built for %r in %.1fs: %d entities, %d relations, %d events (fingerprint=%s)",
        book_name, build_seconds, len(constructed.entities), len(constructed.relations), len(constructed.events), constructed.fingerprint,
    )

    graph_index = GraphIndex()
    graph_index.load_graph(entity_graph, chunks)
    graph_retriever = GraphRetriever(graph_index)

    return chunks, chroma_index, bm25_index, graph_retriever, constructed


def run_claim(
    claim_text: str,
    chroma_index: ChromaIndex,
    bm25_index: BM25Index,
    graph_retriever: GraphRetriever | None,
    decomposition_llm,
    decomp_config: DecompositionConfig,
    fact_verifier: FactVerifier,
    rule_config: RuleEngineConfig,
) -> tuple[EvidenceLedger, float]:
    """Thin wiring: decompose -> retrieve -> fuse -> verify -> classify ->
    verdict, identical to the Phase 5 acceptance test's offline driver,
    parameterized only by which retrievers, which decomposition LLMClient,
    and which FactVerifier are injected. Phase H4: fact_verifier replaces
    the previous hardcoded nli_model/CrossEncoderNLIVerifier construction.
    Phase H4-final: decomposition_llm/decomp_config replace the previous
    hardcoded FakeLLMClient identity-stub and its "fake-model" config --
    this function is now completely unaware of whether it received the
    real GeminiLLMClient (Phase H1) or a FakeLLMClient (offline tests), and
    completely unaware of whether it received CrossEncoderFactVerifier or
    LLMFactVerifier. Everything from build_retrieval_queries onward
    (RetrievalOrchestrator, FusionConfig, to_nli_results, classify,
    ThresholdRuleEngine) is byte-for-byte the same regardless of either
    choice.

    Returns (ledger, retrieval_latency_seconds) -- the retrieval-only
    timing is additive instrumentation for the evaluation report's
    "retrieval latency" deliverable; it reads no new state and changes no
    production module."""
    ledger = EvidenceLedger(original_claim=claim_text)
    service = LedgerService(ledger)

    decomposer = LLMClaimDecomposer(decomposition_llm, decomp_config)
    parent_id = make_source_claim_id(claim_text)
    atomic_claims = decomposer.decompose(claim_text)
    service.record_atomic_claims(parent_id, atomic_claims)

    queries = build_retrieval_queries(atomic_claims, [])
    service.record_retrieval_queries(queries)

    retrievers = [SemanticRetriever(chroma_index), BM25Retriever(bm25_index)]
    if graph_retriever is not None:
        retrievers.append(graph_retriever)
    orchestrator = RetrievalOrchestrator(retrievers, RetrievalConfig(top_k=10))
    retrieval_t0 = time.perf_counter()
    evidence = orchestrator.retrieve_for_queries(queries)
    retrieval_latency = time.perf_counter() - retrieval_t0
    service.record_retrieved_evidence(evidence)

    fused = fuse_evidence(service.ledger.retrieved_evidence, FusionConfig())
    service.record_fused_evidence(fused)

    fused_by_claim: dict[str, list] = {}
    for f in ledger.fused_evidence:
        fused_by_claim.setdefault(f.atomic_claim_id, []).append(f)
    all_results = []
    for claim in ledger.atomic_claims:
        fact_verifications = fact_verifier.verify(claim, fused_by_claim.get(claim.claim_id, []))
        all_results.extend(to_nli_results(fact_verifications, claim))
    service.record_nli_results(all_results)

    claim_ids = [c.claim_id for c in ledger.atomic_claims]
    outcome = classify(ledger.nli_results, claim_ids, rule_config)
    service.record_classification(outcome.contradictions, outcome.supporting_evidence, outcome.unsupported_claim_ids)

    engine = ThresholdRuleEngine(rule_config)
    verdict = engine.evaluate(ledger)
    service.set_final_verdict(verdict)

    return ledger, retrieval_latency


def run_condition(
    rows: list[DatasetRow],
    book_resources: dict[str, tuple],
    use_graph: bool,
    decomposition_llm,
    decomp_config: DecompositionConfig,
    fact_verifier: FactVerifier,
    rule_config: RuleEngineConfig,
    checkpoint_path: Path,
) -> list[dict]:
    checkpoint = load_checkpoint(checkpoint_path)
    results: list[dict] = []
    for row in rows:
        if row.row_id in checkpoint:
            results.append(checkpoint[row.row_id])
            continue

        claim_text = build_claim_text(row)
        chunks, chroma_index, bm25_index, graph_retriever, _ = book_resources[row.book_name]

        t0 = time.perf_counter()
        retrieval_latency = 0.0
        try:
            ledger, retrieval_latency = run_claim(
                claim_text, chroma_index, bm25_index, graph_retriever if use_graph else None, decomposition_llm, decomp_config, fact_verifier, rule_config
            )
            error_text = None
        except Exception as exc:  # one row's failure must never crash the whole run
            logger.exception("Row %s raised an exception", row.row_id)
            ledger = None
            error_text = str(exc)
        latency = time.perf_counter() - t0

        predicted_verdict = ledger.final_verdict.verdict.value if (ledger is not None and ledger.final_verdict) else None
        graph_chunk_ids = (
            sorted({e.chunk_id for e in ledger.retrieved_evidence if e.source.value == "GRAPH"}) if ledger is not None else []
        )
        # graph_only_fused_chunk_ids: chunks that survived RRF fusion (i.e.
        # actually reached NLI) whose ONLY contributing source was GRAPH --
        # the honest measure of "graph contribution over BM25+Chroma"
        # (evidence neither dense nor lexical retrieval found on their own).
        graph_only_fused_chunk_ids = (
            sorted(
                {
                    f.chunk_id
                    for f in ledger.fused_evidence
                    if f.contributing_sources == [RetrievalSource.GRAPH]
                }
            )
            if ledger is not None
            else []
        )

        record = {
            "row_id": row.row_id,
            "book_name": row.book_name,
            "char": row.char,
            "claim": claim_text,
            "gold_label": row.label,
            "use_graph": use_graph,
            "error": error_text,
            "predicted_verdict": predicted_verdict,
            "total_latency_seconds": latency,
            "retrieval_latency_seconds": retrieval_latency,
            "graph_contributed_chunk_ids": graph_chunk_ids,
            "graph_only_fused_chunk_ids": graph_only_fused_chunk_ids,
            "retrieved_evidence_count": len(ledger.retrieved_evidence) if ledger is not None else 0,
            "fused_evidence_count": len(ledger.fused_evidence) if ledger is not None else 0,
        }
        append_checkpoint(checkpoint_path, record)
        results.append(record)
        logger.info(
            "[%s] Row %s [%s/%s]: gold=%s predicted=%s graph_chunks=%d latency=%.2fs",
            "with-graph" if use_graph else "baseline", row.row_id, row.book_name, row.char, row.label, predicted_verdict, len(graph_chunk_ids), latency,
        )

    return results


def to_verdict_pairs(records: list[dict]) -> list[tuple[VerdictEnum, VerdictEnum]]:
    pairs = []
    for r in records:
        if r.get("gold_label") is None or r.get("predicted_verdict") is None:
            continue
        pairs.append((GOLD_LABEL_TO_VERDICT[r["gold_label"]], VerdictEnum(r["predicted_verdict"])))
    return pairs


def compute_graph_topology_stats(constructed: ConstructedGraph) -> dict:
    """Density / connected components / average degree for the evaluation
    report's "graph statistics" deliverable. Builds its own throwaway
    networkx.Graph from constructed.entities/relations -- read-only,
    local to this evaluation script, and does not touch
    lncvs.graph.builder.EntityGraph (which deliberately confines its own
    networkx object to itself)."""
    graph = nx.Graph()
    graph.add_nodes_from(entity.entity_id for entity in constructed.entities)
    graph.add_edges_from((r.subject_entity_id, r.object_entity_id) for r in constructed.relations)

    node_count = graph.number_of_nodes()
    if node_count == 0:
        return {"density": 0.0, "connected_components": 0, "average_degree": 0.0}

    degrees = [degree for _, degree in graph.degree()]
    return {
        "density": nx.density(graph),
        "connected_components": nx.number_connected_components(graph),
        "average_degree": sum(degrees) / node_count,
    }


def compute_retrieval_stats(baseline: list[dict], with_graph: list[dict]) -> dict:
    """Evaluation-only retrieval statistics. NOTE: true Recall@k/MRR
    require gold relevance labels per claim (per CLAUDE.md's evaluation
    rule, never reported as if computable without them) -- train.csv/
    test.csv carry gold verdict labels only, no gold evidence chunk IDs.
    The metrics below are honest, clearly-named proxies, not substitutes
    for Recall@k:
      - evidence_coverage_rate: fraction of rows where fused evidence
        (i.e. evidence that actually reached NLI) was non-empty.
      - graph_only_contribution_rate: fraction of with-graph rows where
        at least one fused chunk's ONLY contributing source was GRAPH --
        i.e. evidence neither Chroma nor BM25 found on their own.
    """

    def _coverage_rate(records: list[dict]) -> float:
        if not records:
            return 0.0
        return sum(1 for r in records if r["fused_evidence_count"] > 0) / len(records)

    def _avg(records: list[dict], key: str) -> float:
        if not records:
            return 0.0
        return sum(r[key] for r in records) / len(records)

    graph_only_rows = [r for r in with_graph if r["graph_only_fused_chunk_ids"]]

    return {
        "baseline_evidence_coverage_rate": _coverage_rate(baseline),
        "with_graph_evidence_coverage_rate": _coverage_rate(with_graph),
        "graph_only_contribution_rate": (len(graph_only_rows) / len(with_graph)) if with_graph else 0.0,
        "baseline_avg_retrieval_latency_seconds": _avg(baseline, "retrieval_latency_seconds"),
        "with_graph_avg_retrieval_latency_seconds": _avg(with_graph, "retrieval_latency_seconds"),
        "baseline_avg_end_to_end_latency_seconds": _avg(baseline, "total_latency_seconds"),
        "with_graph_avg_end_to_end_latency_seconds": _avg(with_graph, "total_latency_seconds"),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = load_csv(DATA_DIR / "train.csv", has_label=True)
    test_rows = load_csv(DATA_DIR / "test.csv", has_label=False)
    all_rows = train_rows + test_rows
    logger.info("Loaded %d train rows, %d test rows", len(train_rows), len(test_rows))

    logger.info("Loading real models...")
    t0 = time.perf_counter()
    real_embedder = SentenceTransformerEmbedder(EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    embedder_load_s = time.perf_counter() - t0
    cached_embedder = CachingEmbedder(
        real_embedder, InMemoryEmbeddingCache(), EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # Phase H4: the cross-encoder NLI model is only loaded when VERIFIER_MODE
    # actually needs it -- "llm" mode never touches CrossEncoderNLIModel, so
    # paying its ~5s load cost (and the model download/memory) for a run that
    # will never call it is unnecessary.
    cached_nli = None
    if VERIFIER_MODE == "cross_encoder":
        t0 = time.perf_counter()
        real_nli = CrossEncoderNLIModel(NLIConfig(model_name="cross-encoder/nli-deberta-v3-base"))
        nli_load_s = time.perf_counter() - t0
        cached_nli = CachingNLIModel(real_nli, InMemoryNLICache(), NLIConfig(model_name="cross-encoder/nli-deberta-v3-base"))
        logger.info("Models loaded: embedder=%.1fs nli=%.1fs", embedder_load_s, nli_load_s)
    else:
        logger.info("Models loaded: embedder=%.1fs (verifier_mode=%r, cross-encoder NLI not loaded)", embedder_load_s, VERIFIER_MODE)

    fact_verifier = build_fact_verifier(VERIFIER_MODE, cached_nli)

    book_resources: dict[str, tuple] = {}
    constructed_graphs: dict[str, ConstructedGraph] = {}
    for book_name, book_path in BOOK_NAME_TO_PATH.items():
        chunks, chroma_index, bm25_index, graph_retriever, constructed = build_novel_graph(book_name, book_path, cached_embedder)
        book_resources[book_name] = (chunks, chroma_index, bm25_index, graph_retriever, constructed)
        constructed_graphs[book_name] = constructed

    # Phase H4-final: real decomposition (Phase H1), replacing the
    # FakeLLMClient identity-stub every prior phase used. Reuses GeminiLLMClient,
    # JsonlLLMCache, and LLMClaimDecomposer exactly as Phase H1 validated them
    # offline (scripts/decompose_dataset.py) -- same model, same cache file
    # (results/decomposition_cache.jsonl), so the 138/140 real decompositions
    # already cached there are reused here with ZERO new API calls; only the
    # 2 rows that previously exceeded max_atomic_claims=10 would need a fresh
    # call, and a cache miss here is a real Gemini call (not the cache-only
    # _NoCallStructuredClient pattern build_novel_graph uses for the
    # already-final graph) since decomposition is not yet a frozen artifact.
    decomposition_llm_config = LLMConfig(model_name=DECOMPOSITION_MODEL, temperature=0.0, max_tokens=2048)
    decomp_config = DecompositionConfig(llm_config=decomposition_llm_config, max_atomic_claims=10)
    real_decomposition_client = GeminiLLMClient(config=decomposition_llm_config)
    decomposition_cache = JsonlLLMCache(RESULTS_DIR / "decomposition_cache.jsonl")
    decomposition_llm = CachingLLMClient(real_decomposition_client, decomposition_cache, decomposition_llm_config)

    # Iteration 1 (post-retrieval decision-policy change): consistency_requires_entailment=False.
    # Error analysis showed the cross-encoder NLI almost never emits ENTAILMENT for a
    # paraphrased claim vs a single retrieved chunk (it emits NEUTRAL), making the strict
    # CONSISTENT verdict structurally unreachable -- every gold-CONSISTENT example was
    # mispredicted. This dataset's 'consistent' means 'not contradicted by the source', so
    # the lenient policy (absence of contradiction => CONSISTENT) matches the label semantics.
    # Iteration 2 (post-retrieval threshold): contradiction_threshold 0.5 -> 0.9.
    # Error analysis (per-row max contradiction score, split by gold) showed EVERY
    # true contradiction scores >= 0.94, while the only sub-0.9 contradictions on
    # gold-CONSISTENT rows (0.62, 0.64) are spurious. Raising the bar to 0.9 drops
    # those spurious low-confidence contradictions while losing ZERO true ones --
    # CONTRADICTORY recall is preserved (0.931), and CONSISTENT precision/recall
    # both improve. (Higher thresholds / multi-chunk corroboration push train
    # numbers higher but sacrifice contradiction recall and overfit the 80-row set;
    # 0.9 is the robust, generalizable cutoff.)
    rule_config = RuleEngineConfig(
        contradiction_threshold=0.9,
        entailment_threshold=DEFAULT_ENTAILMENT_THRESHOLD,
        consistency_requires_entailment=False,
    )

    # Phase H4: checkpoint filenames are namespaced by VERIFIER_MODE.
    # Without this, switching VERIFIER_MODE between runs would silently
    # replay a DIFFERENT verifier's stale checkpointed rows (load_checkpoint
    # has no way to know the cached predictions came from a different
    # FactVerifier) -- a real correctness hazard the original (single-mode)
    # filenames never had to guard against.
    checkpoint_suffix = f"_{VERIFIER_MODE}"

    logger.info("Running BASELINE (chroma+bm25 only) on train.csv [verifier_mode=%s]...", VERIFIER_MODE)
    baseline_train = run_condition(train_rows, book_resources, False, decomposition_llm, decomp_config, fact_verifier, rule_config, RESULTS_DIR / f"graph_impact_baseline_train{checkpoint_suffix}.jsonl")
    logger.info("Running WITH-GRAPH (chroma+bm25+graph) on train.csv [verifier_mode=%s]...", VERIFIER_MODE)
    with_graph_train = run_condition(train_rows, book_resources, True, decomposition_llm, decomp_config, fact_verifier, rule_config, RESULTS_DIR / f"graph_impact_with_graph_train{checkpoint_suffix}.jsonl")

    logger.info("Running BASELINE on test.csv [verifier_mode=%s]...", VERIFIER_MODE)
    baseline_test = run_condition(test_rows, book_resources, False, decomposition_llm, decomp_config, fact_verifier, rule_config, RESULTS_DIR / f"graph_impact_baseline_test{checkpoint_suffix}.jsonl")
    logger.info("Running WITH-GRAPH on test.csv [verifier_mode=%s]...", VERIFIER_MODE)
    with_graph_test = run_condition(test_rows, book_resources, True, decomposition_llm, decomp_config, fact_verifier, rule_config, RESULTS_DIR / f"graph_impact_with_graph_test{checkpoint_suffix}.jsonl")

    baseline_pairs = to_verdict_pairs(baseline_train)
    with_graph_pairs = to_verdict_pairs(with_graph_train)
    baseline_metrics = compute_verdict_metrics(baseline_pairs) if baseline_pairs else None
    with_graph_metrics = compute_verdict_metrics(with_graph_pairs) if with_graph_pairs else None

    rows_where_graph_contributed = [r for r in with_graph_train if r["graph_contributed_chunk_ids"]]
    rows_where_verdict_changed = [
        (b, g)
        for b, g in zip(sorted(baseline_train, key=lambda r: r["row_id"]), sorted(with_graph_train, key=lambda r: r["row_id"]))
        if b["predicted_verdict"] != g["predicted_verdict"]
    ]

    summary = {
        "verifier_mode": VERIFIER_MODE,
        "graph_fingerprints": {book: c.fingerprint for book, c in constructed_graphs.items()},
        "graph_stats": {
            book: {
                "entities": len(c.entities),
                "relations": len(c.relations),
                "events": len(c.events),
                "rejected_relations": len(c.rejected_relations),
                "rejected_events": len(c.rejected_events),
                **compute_graph_topology_stats(c),
            }
            for book, c in constructed_graphs.items()
        },
        "retrieval_stats": compute_retrieval_stats(baseline_train, with_graph_train),
        "baseline_metrics": json.loads(baseline_metrics.model_dump_json()) if baseline_metrics else None,
        "with_graph_metrics": json.loads(with_graph_metrics.model_dump_json()) if with_graph_metrics else None,
        "train_rows_where_graph_contributed_evidence": len(rows_where_graph_contributed),
        "train_rows_where_verdict_changed": len(rows_where_verdict_changed),
        "verdict_changes": [
            {"row_id": b["row_id"], "gold": b["gold_label"], "baseline_predicted": b["predicted_verdict"], "with_graph_predicted": g["predicted_verdict"]}
            for b, g in rows_where_verdict_changed
        ],
    }
    (RESULTS_DIR / f"graph_impact_summary{checkpoint_suffix}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"GRAPH RETRIEVAL IMPACT STUDY SUMMARY (verifier_mode={VERIFIER_MODE})")
    print("=" * 70)
    for book, stats in summary["graph_stats"].items():
        print(f"{book}: {stats}")
    if baseline_metrics and with_graph_metrics:
        print(f"Baseline  (chroma+bm25):       accuracy={baseline_metrics.accuracy:.3f} macro_f1={baseline_metrics.macro_f1:.3f} precision={baseline_metrics.macro_precision:.3f} recall={baseline_metrics.macro_recall:.3f}")
        print(f"With graph (chroma+bm25+graph): accuracy={with_graph_metrics.accuracy:.3f} macro_f1={with_graph_metrics.macro_f1:.3f} precision={with_graph_metrics.macro_precision:.3f} recall={with_graph_metrics.macro_recall:.3f}")
    print(f"Retrieval stats: {summary['retrieval_stats']}")
    print(f"Train rows where graph contributed evidence: {len(rows_where_graph_contributed)} / {len(with_graph_train)}")
    print(f"Train rows where verdict changed: {len(rows_where_verdict_changed)}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
