"""Phase H4-final smoke test: confirms the complete real pipeline

    Backstory
      |
    LLMClaimDecomposer (real, Phase H1)
      |
    AtomicClaims
      |
    Hybrid Retrieval (Chroma + BM25 + Graph)
      |
    FactVerifier (CrossEncoderFactVerifier, Phase H2 -- VERIFIER_MODE
                  defaults to "cross_encoder" in evaluate_with_graph.py,
                  zero API cost, since LLMFactVerifier's live demo
                  remains blocked on Gemini credits per Phase H3)
      |
    Rule Engine (frozen ThresholdRuleEngine)

on ~10 real rows, end to end, with REAL decomposition (reusing
results/decomposition_cache.jsonl -- 138/140 rows are already cached from
Phase H1's full-dataset run, so this incurs ~zero new API cost).

Standalone, additive script -- does not modify evaluate_with_graph.py.
Reuses build_novel_graph and build_fact_verifier from it directly so the
graph/verifier construction is identical to the real eval driver, but
manually re-runs the decompose->retrieve->fuse->verify->classify->verdict
sequence inline (rather than calling run_claim) so it can print the
intermediate FactVerification objects run_claim discards after converting
them to NLIResult -- exactly what this smoke test needs to report and
run_claim has no reason to expose in production.
"""

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

from evaluate_dataset import BOOK_NAME_TO_PATH, build_claim_text, load_csv  # noqa: E402
from evaluate_with_graph import RESULTS_DIR, build_fact_verifier, build_novel_graph  # noqa: E402

from lncvs.fusion import FusionConfig, fuse_evidence  # noqa: E402
from lncvs.indexing import CachingEmbedder, EmbeddingConfig, InMemoryEmbeddingCache, SentenceTransformerEmbedder  # noqa: E402
from lncvs.ledger import LedgerService  # noqa: E402
from lncvs.llm import CachingLLMClient, GeminiLLMClient, JsonlLLMCache, LLMConfig  # noqa: E402
from lncvs.reasoning.decomposition import DecompositionConfig, LLMClaimDecomposer, make_source_claim_id  # noqa: E402
from lncvs.reasoning.fact_verification import to_nli_results  # noqa: E402
from lncvs.reasoning.nli import CachingNLIModel, CrossEncoderNLIModel, InMemoryNLICache, NLIConfig  # noqa: E402
from lncvs.retrieval import BM25Retriever, RetrievalConfig, RetrievalOrchestrator, SemanticRetriever, build_retrieval_queries  # noqa: E402
from lncvs.rules import RuleEngineConfig, ThresholdRuleEngine, classify  # noqa: E402
from lncvs.schemas import EvidenceLedger  # noqa: E402

import logging  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test_h4_final")

DATA_DIR = REPO_ROOT / "data"
DECOMPOSITION_MODEL = "gemini-2.5-flash"
N_SMOKE_ROWS = 10


def main() -> int:
    train_rows = load_csv(DATA_DIR / "train.csv", has_label=True)
    smoke_rows = train_rows[:N_SMOKE_ROWS]
    print(f"Smoke-testing {len(smoke_rows)} rows.\n")

    real_embedder = SentenceTransformerEmbedder(EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    cached_embedder = CachingEmbedder(real_embedder, InMemoryEmbeddingCache(), EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    real_nli = CrossEncoderNLIModel(NLIConfig(model_name="cross-encoder/nli-deberta-v3-base"))
    cached_nli = CachingNLIModel(real_nli, InMemoryNLICache(), NLIConfig(model_name="cross-encoder/nli-deberta-v3-base"))
    fact_verifier = build_fact_verifier("cross_encoder", cached_nli)

    needed_books = {row.book_name for row in smoke_rows}
    book_resources = {}
    for book_name in needed_books:
        book_path = BOOK_NAME_TO_PATH[book_name]
        chunks, chroma_index, bm25_index, graph_retriever, _ = build_novel_graph(book_name, book_path, cached_embedder)
        book_resources[book_name] = (chunks, chroma_index, bm25_index, graph_retriever)

    decomposition_llm_config = LLMConfig(model_name=DECOMPOSITION_MODEL, temperature=0.0, max_tokens=2048)
    decomp_config = DecompositionConfig(llm_config=decomposition_llm_config, max_atomic_claims=10)
    real_decomposition_client = GeminiLLMClient(config=decomposition_llm_config)
    decomposition_cache = JsonlLLMCache(RESULTS_DIR / "decomposition_cache.jsonl")
    decomposition_llm = CachingLLMClient(real_decomposition_client, decomposition_cache, decomposition_llm_config)
    decomposer = LLMClaimDecomposer(decomposition_llm, decomp_config)

    rule_config = RuleEngineConfig(contradiction_threshold=0.9, entailment_threshold=0.5, consistency_requires_entailment=False)

    for i, row in enumerate(smoke_rows, start=1):
        claim_text = build_claim_text(row)
        chunks, chroma_index, bm25_index, graph_retriever = book_resources[row.book_name]

        print("=" * 80)
        print(f"ROW {i}/{len(smoke_rows)}  id={row.row_id}  book={row.book_name}  char={row.char}  gold={row.label}")
        print(f"BACKSTORY: {claim_text}")

        ledger = EvidenceLedger(original_claim=claim_text)
        service = LedgerService(ledger)

        try:
            parent_id = make_source_claim_id(claim_text)
            atomic_claims = decomposer.decompose(claim_text)
        except ValueError as exc:
            print(f"  DECOMPOSITION FAILED: {exc}")
            continue
        service.record_atomic_claims(parent_id, atomic_claims)
        print(f"ATOMIC CLAIMS ({len(atomic_claims)}):")
        for claim in atomic_claims:
            print(f"  - {claim.text}")

        queries = build_retrieval_queries(atomic_claims, [])
        service.record_retrieval_queries(queries)

        retrievers = [SemanticRetriever(chroma_index), BM25Retriever(bm25_index), graph_retriever]
        orchestrator = RetrievalOrchestrator(retrievers, RetrievalConfig(top_k=10))
        evidence = orchestrator.retrieve_for_queries(queries)
        service.record_retrieved_evidence(evidence)

        fused = fuse_evidence(service.ledger.retrieved_evidence, FusionConfig())
        service.record_fused_evidence(fused)

        fused_by_claim: dict[str, list] = {}
        for f in ledger.fused_evidence:
            fused_by_claim.setdefault(f.atomic_claim_id, []).append(f)

        print("RETRIEVAL + VERIFICATION PER ATOMIC CLAIM:")
        all_results = []
        for claim in atomic_claims:
            claim_evidence = fused_by_claim.get(claim.claim_id, [])
            print(f"  [{claim.text!r}] retrieved {len(claim_evidence)} fused evidence record(s)")
            fact_verifications = fact_verifier.verify(claim, claim_evidence)
            for fv in fact_verifications:
                print(f"      chunk={fv.evidence_chunk_id!r} label={fv.label.value} confidence={fv.confidence:.3f}")
                print(f"          quotes={fv.supporting_quotes!r}")
                print(f"          explanation={fv.explanation!r}")
            all_results.extend(to_nli_results(fact_verifications, claim))
        service.record_nli_results(all_results)

        claim_ids = [c.claim_id for c in ledger.atomic_claims]
        outcome = classify(ledger.nli_results, claim_ids, rule_config)
        service.record_classification(outcome.contradictions, outcome.supporting_evidence, outcome.unsupported_claim_ids)

        engine = ThresholdRuleEngine(rule_config)
        verdict = engine.evaluate(ledger)
        service.set_final_verdict(verdict)

        print(f"FINAL VERDICT: {verdict.verdict.value}  (fired_rule={verdict.fired_rule}, gold={row.label})")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
