"""Post-retrieval threshold error-analysis harness (no API calls, no graph
rebuild beyond cache replay).

NLI results are INDEPENDENT of the rule-engine thresholds -- only
lncvs.rules.classification.classify() applies thresholds, downstream of NLI.
So we run NLI exactly once per train row (graphs replay from the cached
extractions, zero Gemini calls), persist every (atomic_claim_id, label,
score), and then sweep (contradiction_threshold, entailment_threshold)
analytically over the persisted scores -- recomputing verdicts with the
real classify()+ThresholdRuleEngine rule logic, never re-running NLI.

This lets us find the single highest-impact threshold change in one pass
instead of burning many full evaluation runs guessing.

Reuses build_novel_graph / run_claim plumbing conceptually but captures the
ledger's nli_results instead of only the verdict. Touches nothing before
retrieval.
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
    GOLD_LABEL_TO_VERDICT,
    build_claim_text,
    build_decomposition_llm_for_rows,
    load_csv,
)

from lncvs.chunking import ChunkingConfig, chunk_document  # noqa: E402
from lncvs.fusion import FusionConfig, fuse_evidence  # noqa: E402
from lncvs.graph import GraphIndex, GraphRetriever  # noqa: E402
from lncvs.graph.construction import build_graph_for_novel  # noqa: E402
from lncvs.graph.llm_extraction import ExtractionConfig, LLMWindowExtractor  # noqa: E402
from lncvs.indexing import BM25Index, CachingEmbedder, ChromaIndex, EmbeddingConfig, InMemoryEmbeddingCache, SentenceTransformerEmbedder  # noqa: E402
from lncvs.ingestion import load_and_clean_narrative  # noqa: E402
from lncvs.ledger import LedgerService  # noqa: E402
from lncvs.llm import CachingStructuredLLMClient, JsonlStructuredLLMCache, LLMConfig  # noqa: E402
from lncvs.reasoning.decomposition import DecompositionConfig, LLMClaimDecomposer, make_source_claim_id  # noqa: E402
from lncvs.reasoning.nli import CachingNLIModel, CrossEncoderNLIModel, CrossEncoderNLIVerifier, InMemoryNLICache, NLIConfig  # noqa: E402
from lncvs.retrieval import BM25Retriever, RetrievalConfig, RetrievalOrchestrator, SemanticRetriever, build_retrieval_queries  # noqa: E402
from lncvs.schemas import EvidenceLedger, NLILabel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("analyze_thresholds")

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
NLI_DUMP_PATH = RESULTS_DIR / "threshold_analysis_nli_dump.json"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
EXTRACTION_MODEL = "gemini-2.5-flash"


class _NoCallStructuredClient:
    """A StructuredLLMClient that raises if ever actually invoked. Wrapped
    behind CachingStructuredLLMClient, a full cache hit means it is never
    called -- proving this harness makes ZERO Gemini API calls. A miss
    fails loudly rather than silently hitting the network (there is no key
    anyway), which is exactly the desired behavior under the 'graph is
    final and read-only' constraint."""

    def complete_structured(self, prompt, response_schema):
        # ValueError (not RuntimeError) so the pipeline's existing per-window
        # fault-isolation (construction/pipeline.py catches ValueError) skips
        # this window -- which is exactly what happened in the original run
        # for the one window that returned no content (Monte Cristo chapter
        # 156) and was therefore never cached. Skipping it here reproduces
        # the IDENTICAL graph. Any cache hit means this is never reached, so
        # zero Gemini API calls are made regardless.
        raise ValueError(
            "Cache miss during threshold analysis (window not in extraction cache -- "
            "skipped to reproduce the original graph exactly; no Gemini API call made)."
        )


def build_cached_graph(book_name, book_path, cached_embedder):
    """Identical to evaluate_with_graph.build_novel_graph but with the real
    Gemini client swapped for _NoCallStructuredClient -- builds Chroma+BM25
    +graph entirely from the on-disk extraction cache, zero API calls."""
    document = load_and_clean_narrative(book_path, source_id=book_name)
    chunks = chunk_document(document, ChunkingConfig(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP))
    safe_name = book_name.replace(" ", "_").lower()
    chroma_index = ChromaIndex(embedder=cached_embedder, collection_name=f"thresh-{safe_name}")
    chroma_index.index(chunks)
    bm25_index = BM25Index(collection_name=f"thresh-{safe_name}-bm25")
    bm25_index.index(chunks)

    llm_config = LLMConfig(model_name=EXTRACTION_MODEL, temperature=0.0, max_tokens=65536)
    extraction_config = ExtractionConfig()
    cache = JsonlStructuredLLMCache(RESULTS_DIR / f"extraction_cache_{safe_name}.jsonl")
    caching_client = CachingStructuredLLMClient(_NoCallStructuredClient(), cache, llm_config, extraction_config.schema_version)
    extractor = LLMWindowExtractor(caching_client, extraction_config)

    _, entity_graph = build_graph_for_novel(document.cleaned_text, chunks, extractor)
    graph_index = GraphIndex()
    graph_index.load_graph(entity_graph, chunks)
    return chroma_index, bm25_index, GraphRetriever(graph_index)


def collect_nli_for_row(claim_text, chroma_index, bm25_index, graph_retriever, decomposition_llm, nli_model):
    """Run decompose -> retrieve(graph included) -> fuse -> NLI and return the
    per-claim NLI results, WITHOUT applying any threshold/verdict. Mirrors
    evaluate_with_graph.run_claim up to NLI, then stops."""
    ledger = EvidenceLedger(original_claim=claim_text)
    service = LedgerService(ledger)

    decomp_config = DecompositionConfig(llm_config=LLMConfig(model_name="fake-model"))
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
    evidence = orchestrator.retrieve_for_queries(queries)
    service.record_retrieved_evidence(evidence)

    fused = fuse_evidence(service.ledger.retrieved_evidence, FusionConfig())
    service.record_fused_evidence(fused)

    verifier = CrossEncoderNLIVerifier(nli_model)
    fused_by_claim = {}
    for f in ledger.fused_evidence:
        fused_by_claim.setdefault(f.atomic_claim_id, []).append(f)
    all_results = []
    for claim in ledger.atomic_claims:
        all_results.extend(verifier.verify(claim, fused_by_claim.get(claim.claim_id, [])))
    service.record_nli_results(all_results)

    claim_ids = [c.claim_id for c in ledger.atomic_claims]
    per_claim = {cid: [] for cid in claim_ids}
    for r in ledger.nli_results:
        per_claim.setdefault(r.atomic_claim_id, []).append({"label": r.label.value, "score": r.score})
    return claim_ids, per_claim


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = load_csv(DATA_DIR / "train.csv", has_label=True)
    logger.info("Loaded %d train rows", len(train_rows))

    real_embedder = SentenceTransformerEmbedder(EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    cached_embedder = CachingEmbedder(
        real_embedder, InMemoryEmbeddingCache(), EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    real_nli = CrossEncoderNLIModel(NLIConfig(model_name="cross-encoder/nli-deberta-v3-base"))
    cached_nli = CachingNLIModel(real_nli, InMemoryNLICache(), NLIConfig(model_name="cross-encoder/nli-deberta-v3-base"))

    book_resources = {}
    for book_name, book_path in BOOK_NAME_TO_PATH.items():
        chroma_index, bm25_index, graph_retriever = build_cached_graph(book_name, book_path, cached_embedder)
        book_resources[book_name] = (chroma_index, bm25_index, graph_retriever)

    decomposition_llm = build_decomposition_llm_for_rows(train_rows)

    dump = []
    for i, row in enumerate(train_rows):
        claim_text = build_claim_text(row)
        chroma_index, bm25_index, graph_retriever = book_resources[row.book_name]
        t0 = time.perf_counter()
        claim_ids, per_claim = collect_nli_for_row(
            claim_text, chroma_index, bm25_index, graph_retriever, decomposition_llm, cached_nli
        )
        dump.append({
            "row_id": row.row_id,
            "book_name": row.book_name,
            "gold_label": row.label,
            "claim_ids": claim_ids,
            "nli_by_claim": per_claim,
        })
        logger.info("[%d/%d] row %s: %d atomic claims, %.2fs", i + 1, len(train_rows), row.row_id, len(claim_ids), time.perf_counter() - t0)

    NLI_DUMP_PATH.write_text(json.dumps(dump, indent=2), encoding="utf-8")
    logger.info("Wrote NLI dump for %d rows to %s", len(dump), NLI_DUMP_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
