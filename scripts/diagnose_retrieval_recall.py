"""Retrieval-recall diagnostic (follow-up to H4.5): measures Recall@k/MRR
on a small, hand-verified GoldSpan mini-set, comparing retrieval WITH vs
WITHOUT probe questions (lncvs.reasoning.questions.LLMQuestionGenerator).

Motivation (from the H4.5 diagnosis): neither CrossEncoderFactVerifier nor
LLMFactVerifier could be fairly compared because the evidence retrieved
for several gold-contradict rows was, on inspection, irrelevant -- RRF
scores at the floor, no lexical or semantic connection to the claim's
actual content. evaluate_with_graph.run_claim calls
build_retrieval_queries(atomic_claims, []) -- the probe_questions argument
is hardcoded empty, so the built, frozen Question Generation module
(reasoning/questions/) is never exercised in that script. This is a
read-only diagnostic: it does NOT modify evaluate_with_graph.py, retrieval,
fusion, or any frozen component -- it only adds probe questions as
ADDITIONAL retrieval queries (the documented, additive role
build_retrieval_queries already supports) for comparison purposes.

GoldSpan provenance (hand-verified against the real source text, not
guessed):
- Row 1 claim "Jacques Paganel fell in love with geography." -> the
  novel's actual Paganel introduction states he spent "twenty years of
  his life in studying geography" (char offset confirmed via
  load_and_clean_narrative). A DIRECT, lexically-close case.
- Row 11 claim "Jacques Paganel was slashed across the left forearm with
  a dagger." (embedded in a fabricated Ayrton/slave-traders anecdote) ->
  the real, INDIRECT answer is the novel's reveal that Ayrton is actually
  "Ben Joyce," leader of an escaped-convict gang -- confirmed via the
  real "commanded by a certain Ben Joyce" passage. This is a vocabulary-
  mismatch case (no shared words with the claim at all) included
  specifically to test whether probe-question reformulation can bridge
  it, since plain semantic/lexical retrieval on the claim text alone has
  no chance of finding it.
"""

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from evaluate_dataset import BOOK_NAME_TO_PATH, load_csv  # noqa: E402
from evaluate_with_graph import build_novel_graph  # noqa: E402

from lncvs.evaluation.dataset import map_spans_to_chunks  # noqa: E402
from lncvs.evaluation.metrics.retrieval import compute_retrieval_metrics  # noqa: E402
from lncvs.fusion import FusionConfig, fuse_evidence  # noqa: E402
from lncvs.indexing import CachingEmbedder, EmbeddingConfig, InMemoryEmbeddingCache, SentenceTransformerEmbedder  # noqa: E402
from lncvs.ledger import LedgerService  # noqa: E402
from lncvs.llm import CachingLLMClient, JsonlLLMCache, LLMConfig  # noqa: E402
from lncvs.reasoning.questions import LLMQuestionGenerator, QuestionGenerationConfig  # noqa: E402
from lncvs.retrieval import BM25Retriever, RetrievalConfig, RetrievalOrchestrator, SemanticRetriever, build_retrieval_queries  # noqa: E402
from lncvs.schemas import AtomicClaim, EvidenceLedger, GoldSpan  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("diagnose_retrieval_recall")

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
BOOK_NAME = "In Search of the Castaways"
K_CUTOFFS = [5, 10, 20]

# Hand-verified gold spans, see module docstring for the textual evidence
# behind each. claim_text must match the exact decomposed atomic claim
# text already present in results/decomposition_cache.jsonl (Phase H1) so
# this script makes zero new decomposition calls.
GOLD_CASES = [
    {
        "label": "row1_paganel_geography (DIRECT)",
        "claim_text": "Jacques Paganel fell in love with geography.",
        "gold_spans": [GoldSpan(char_start=76002, char_end=76200, note="Paganel intro: '...passing twenty years of his life in studying geography...'")],
    },
    {
        "label": "row11_paganel_dagger (INDIRECT, vocabulary mismatch)",
        "claim_text": "Jacques Paganel was slashed across the left forearm with a dagger.",
        "gold_spans": [GoldSpan(char_start=466400, char_end=466800, note="Ayrton revealed as 'Ben Joyce', leader of an escaped-convict gang -- the real answer to the fabricated slave-traders/dagger anecdote, via completely different vocabulary.")],
    },
]


def main() -> int:
    logger.setLevel(logging.INFO)
    real_embedder = SentenceTransformerEmbedder(EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    cached_embedder = CachingEmbedder(real_embedder, InMemoryEmbeddingCache(), EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))

    chunks, chroma_index, bm25_index, graph_retriever, _ = build_novel_graph(BOOK_NAME, BOOK_NAME_TO_PATH[BOOK_NAME], cached_embedder)
    logger.info("Indexed %d chunks for %r", len(chunks), BOOK_NAME)

    # Question generation: real OpenAI calls (gpt-4o-mini, cheap, cached).
    # Additive only -- evaluate_with_graph.py and its VERIFICATION_MODEL/
    # VERIFIER_MODE constants are untouched; this script builds its own
    # generator locally, exactly as validate_h5_sample.py did for the
    # OpenAI-backed LLMFactVerifier. LLMQuestionGenerator wants a plain
    # LLMClient (text completion); no plain-text OpenAI adapter exists yet
    # in lncvs.llm (only OpenAIStructuredClient), so this script defines a
    # small local one -- diagnostic-only, not a production addition.
    from lncvs.llm.base import LLMCompletion

    class _OpenAITextClient:
        def __init__(self, config: LLMConfig) -> None:
            import openai

            self._client = openai.OpenAI()
            self._config = config

        def complete(self, prompt: str) -> LLMCompletion:
            response = self._client.chat.completions.create(
                model=self._config.model_name,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return LLMCompletion(text=response.choices[0].message.content, model_fingerprint=self._config.fingerprint())

    question_llm_config = LLMConfig(model_name="gpt-4o-mini", temperature=0.0, max_tokens=512)
    question_config = QuestionGenerationConfig(llm_config=question_llm_config, max_questions_per_claim=10)
    question_cache = JsonlLLMCache(RESULTS_DIR / "question_generation_cache.jsonl")
    question_llm = CachingLLMClient(_OpenAITextClient(question_llm_config), question_cache, question_llm_config)
    question_generator = LLMQuestionGenerator(question_llm, question_config)

    print("\n" + "=" * 70)
    print("RETRIEVAL RECALL DIAGNOSTIC: WITHOUT vs WITH probe questions")
    print("=" * 70)

    for case in GOLD_CASES:
        gold_chunk_ids = map_spans_to_chunks(case["gold_spans"], chunks)
        print(f"\n--- {case['label']} ---")
        print(f"Claim: {case['claim_text']!r}")
        print(f"Gold chunk_ids ({len(gold_chunk_ids)}): {sorted(gold_chunk_ids)}")
        if not gold_chunk_ids:
            print("WARNING: gold span did not overlap any chunk -- skipping (cannot compute Recall@k).")
            continue

        atomic_claim = AtomicClaim(claim_id=_claim_id_for(case["claim_text"]), text=case["claim_text"])

        for use_probe_questions in (False, True):
            queries = build_retrieval_queries(
                [atomic_claim],
                question_generator.generate(atomic_claim) if use_probe_questions else [],
            )
            retrievers = [SemanticRetriever(chroma_index), BM25Retriever(bm25_index), graph_retriever]
            orchestrator = RetrievalOrchestrator(retrievers, RetrievalConfig(top_k=10))
            evidence = orchestrator.retrieve_for_queries(queries)
            fused = fuse_evidence(evidence, FusionConfig())

            ledger = EvidenceLedger(original_claim=case["claim_text"])
            service = LedgerService(ledger)
            service.record_atomic_claims("diagnostic-parent", [atomic_claim])
            service.record_retrieval_queries(queries)
            service.record_retrieved_evidence(evidence)
            service.record_fused_evidence(fused)

            metrics = compute_retrieval_metrics(ledger, gold_chunk_ids, K_CUTOFFS)
            label = "WITH probe questions" if use_probe_questions else "WITHOUT probe questions (baseline)"
            n_queries = len(queries)
            if metrics is None:
                print(f"  {label}: n_queries={n_queries} -- no gold chunk_ids (should not happen here)")
            else:
                cutoffs_str = ", ".join(f"R@{c.k}={c.recall:.2f}/P@{c.k}={c.precision:.2f}" for c in metrics.cutoffs)
                print(f"  {label}: n_queries={n_queries} MRR={metrics.mrr:.3f} {cutoffs_str}")
                if use_probe_questions:
                    print(f"    Probe questions generated: {[q.text for q in question_generator.generate(atomic_claim)]}")

    print("\n" + "=" * 70 + "\n")
    return 0


def _claim_id_for(claim_text: str) -> str:
    import hashlib

    return hashlib.sha256(claim_text.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    sys.exit(main())
