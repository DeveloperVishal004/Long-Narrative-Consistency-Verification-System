"""Read-only diagnostic (H8): for every gold-CONTRADICTORY row the LLM
verifier (H7, baseline condition) predicted CONSISTENT for, determine
whether the failure is a RETRIEVAL miss (no fused evidence chunk for the
relevant atomic claim ever came from the book at all / very little fused
evidence) or a VERIFIER miss (evidence was fused and reached the verifier,
but it returned NOT_MENTIONED/ENTAILED rather than CONTRADICTED).

Makes ZERO new LLM API calls: reuses the exact same cached
decomposition_llm (results/decomposition_cache.jsonl) and cached
fact_verifier (results/fact_verification_cache_openai_gpt_4o_2024_08_06.jsonl)
construction as scripts/evaluate_with_graph.py's baseline condition, so every
call is a guaranteed cache hit for rows already checkpointed there.

This script does not modify any production module, any checkpoint file, or
any cache file -- it only reads existing caches and prints a report.
"""

import json
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

from evaluate_dataset import BOOK_NAME_TO_PATH, build_claim_text  # noqa: E402
from evaluate_with_graph import (  # noqa: E402
    DECOMPOSITION_MODEL,
    RESULTS_DIR,
    build_fact_verifier,
    build_novel_graph,
)
from lncvs.indexing import CachingEmbedder, EmbeddingConfig, InMemoryEmbeddingCache, SentenceTransformerEmbedder  # noqa: E402
from lncvs.ledger import LedgerService  # noqa: E402
from lncvs.llm import CachingLLMClient, GeminiLLMClient, JsonlLLMCache, LLMConfig  # noqa: E402
from lncvs.reasoning.decomposition import DecompositionConfig, LLMClaimDecomposer, make_source_claim_id  # noqa: E402
from lncvs.reasoning.fact_verification import to_nli_results  # noqa: E402
from lncvs.fusion import FusionConfig, fuse_evidence  # noqa: E402
from lncvs.retrieval import BM25Retriever, RetrievalConfig, RetrievalOrchestrator, SemanticRetriever, build_retrieval_queries  # noqa: E402
from lncvs.schemas import EvidenceLedger  # noqa: E402


def main() -> int:
    baseline_rows = [json.loads(l) for l in open(RESULTS_DIR / "graph_impact_baseline_train_llm.jsonl")]
    missed = [r for r in baseline_rows if r["gold_label"] == "contradict" and r["predicted_verdict"] == "CONSISTENT"]
    print(f"Diagnosing {len(missed)} missed-contradiction rows (gold=contradict, predicted=CONSISTENT)\n")

    real_embedder = SentenceTransformerEmbedder(EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    cached_embedder = CachingEmbedder(real_embedder, InMemoryEmbeddingCache(), EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))

    book_resources = {}
    for book_name, book_path in BOOK_NAME_TO_PATH.items():
        chunks, chroma_index, bm25_index, graph_retriever, constructed = build_novel_graph(book_name, book_path, cached_embedder)
        book_resources[book_name] = (chunks, chroma_index, bm25_index, graph_retriever, constructed)

    decomposition_llm_config = LLMConfig(model_name=DECOMPOSITION_MODEL, temperature=0.0, max_tokens=2048)
    decomp_config = DecompositionConfig(llm_config=decomposition_llm_config, max_atomic_claims=10)
    real_decomposition_client = GeminiLLMClient(config=decomposition_llm_config)
    decomposition_cache = JsonlLLMCache(RESULTS_DIR / "decomposition_cache.jsonl")
    decomposition_llm = CachingLLMClient(real_decomposition_client, decomposition_cache, decomposition_llm_config)

    fact_verifier = build_fact_verifier("llm", None)

    retrieval_miss = 0
    verifier_miss = 0
    details = []

    for row in missed:
        claim_text = build_claim_text(type("R", (), {
            "char": row["char"], "book_name": row["book_name"], "claim": row["claim"]
        })()) if False else row["claim"]
        chunks, chroma_index, bm25_index, graph_retriever, _ = book_resources[row["book_name"]]

        ledger = EvidenceLedger(original_claim=claim_text)
        service = LedgerService(ledger)
        decomposer = LLMClaimDecomposer(decomposition_llm, decomp_config)
        parent_id = make_source_claim_id(claim_text)
        atomic_claims = decomposer.decompose(claim_text)
        service.record_atomic_claims(parent_id, atomic_claims)

        queries = build_retrieval_queries(atomic_claims, [])
        service.record_retrieval_queries(queries)
        retrievers = [SemanticRetriever(chroma_index), BM25Retriever(bm25_index)]
        orchestrator = RetrievalOrchestrator(retrievers, RetrievalConfig(top_k=10))
        evidence = orchestrator.retrieve_for_queries(queries)
        service.record_retrieved_evidence(evidence)

        fused = fuse_evidence(service.ledger.retrieved_evidence, FusionConfig())
        service.record_fused_evidence(fused)

        fused_by_claim = {}
        for f in ledger.fused_evidence:
            fused_by_claim.setdefault(f.atomic_claim_id, []).append(f)

        row_has_verifier_evidence = False
        row_verdicts = []
        for claim in ledger.atomic_claims:
            ev = fused_by_claim.get(claim.claim_id, [])
            fact_verifications = fact_verifier.verify(claim, ev)
            labels = {fv.label.value for fv in fact_verifications}
            explanation = fact_verifications[0].explanation[:150] if fact_verifications else ""
            row_verdicts.append((claim.text[:80], len(ev), sorted(labels), explanation))
            if len(ev) > 0:
                row_has_verifier_evidence = True

        no_evidence_at_all = all(v[1] == 0 for v in row_verdicts) if row_verdicts else True

        if no_evidence_at_all:
            retrieval_miss += 1
            category = "RETRIEVAL_MISS (zero fused evidence for every atomic claim)"
        else:
            verifier_miss += 1
            category = "VERIFIER_MISS (evidence was fused/reached verifier, but no CONTRADICTION fired)"

        details.append((row["row_id"], row["book_name"], row["char"], category, row_verdicts))

    print("=" * 70)
    print(f"RETRIEVAL_MISS: {retrieval_miss} / {len(missed)}")
    print(f"VERIFIER_MISS:  {verifier_miss} / {len(missed)}")
    print("=" * 70)
    for row_id, book, char, category, verdicts in details:
        print(f"\nRow {row_id} [{book} / {char}]: {category}")
        for claim_text, n_ev, label, expl in verdicts:
            print(f"    claim={claim_text!r} fused_evidence={n_ev} verifier_label={label} explanation={expl!r}")

    (RESULTS_DIR / "missed_contradiction_diagnosis.json").write_text(
        json.dumps(
            {
                "n_missed": len(missed),
                "retrieval_miss": retrieval_miss,
                "verifier_miss": verifier_miss,
                "details": [
                    {"row_id": r, "book": b, "char": c, "category": cat, "verdicts": v}
                    for r, b, c, cat, v in details
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
