"""Slice H1 verification script: activates the real LLMClaimDecomposer
(replacing the identity-stub FakeLLMClient used everywhere else so far)
and runs it over every real backstory claim in train.csv + test.csv.

Scope, deliberately narrow per Slice H1: Backstory -> AtomicClaim list,
verified on real examples. Does NOT touch retrieval, NLI, or the rule
engine -- this script makes no calls into lncvs.retrieval, lncvs.fusion,
lncvs.reasoning.nli, or lncvs.rules at all. Wiring real decomposition into
the full claim-to-verdict pipeline is a later slice (H4).

Reuses, unmodified: LLMClaimDecomposer, DecompositionConfig, AtomicClaim,
make_source_claim_id, render_decomposition_prompt, CachingLLMClient. The
only new pieces (Phase H1) are GeminiLLMClient (a real LLMClient
implementation) and JsonlLLMCache (persistent caching for that protocol,
mirroring the existing JsonlStructuredLLMCache discipline).
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

from evaluate_dataset import build_claim_text, load_csv  # noqa: E402

from lncvs.llm import CachingLLMClient, GeminiLLMClient, JsonlLLMCache, LLMConfig  # noqa: E402
from lncvs.reasoning.decomposition import DecompositionConfig, LLMClaimDecomposer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("decompose_dataset")

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
DECOMPOSITION_MODEL = "gemini-2.5-flash"


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = load_csv(DATA_DIR / "train.csv", has_label=True)
    test_rows = load_csv(DATA_DIR / "test.csv", has_label=False)
    all_rows = train_rows + test_rows
    logger.info("Loaded %d train rows, %d test rows", len(train_rows), len(test_rows))

    llm_config = LLMConfig(model_name=DECOMPOSITION_MODEL, temperature=0.0, max_tokens=2048)
    decomp_config = DecompositionConfig(llm_config=llm_config, max_atomic_claims=10)

    real_client = GeminiLLMClient(config=llm_config)
    cache = JsonlLLMCache(RESULTS_DIR / "decomposition_cache.jsonl")
    caching_client = CachingLLMClient(real_client, cache, llm_config)
    decomposer = LLMClaimDecomposer(caching_client, decomp_config)

    dump = []
    n_facts_total = 0
    n_failed = 0
    t0 = time.perf_counter()
    for i, row in enumerate(all_rows):
        claim_text = build_claim_text(row)
        try:
            atomic_claims = decomposer.decompose(claim_text)
        except ValueError as exc:
            logger.warning("Row %s decomposition failed: %s", row.row_id, exc)
            n_failed += 1
            continue

        n_facts_total += len(atomic_claims)
        dump.append({
            "row_id": row.row_id,
            "book_name": row.book_name,
            "char": row.char,
            "original_claim": claim_text,
            "atomic_claims": [{"claim_id": c.claim_id, "text": c.text, "index": c.index} for c in atomic_claims],
        })
        logger.info("[%d/%d] row %s (%s): %d atomic facts", i + 1, len(all_rows), row.row_id, row.char, len(atomic_claims))

    elapsed = time.perf_counter() - t0
    out_path = RESULTS_DIR / "decomposition_dump.json"
    out_path.write_text(json.dumps(dump, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("ATOMIC CLAIM DECOMPOSITION -- SLICE H1 VERIFICATION")
    print("=" * 70)
    print(f"Rows processed:      {len(dump)} / {len(all_rows)} ({n_failed} failed)")
    print(f"Total atomic facts:  {n_facts_total}")
    print(f"Mean facts/row:      {n_facts_total / len(dump):.2f}" if dump else "Mean facts/row:      n/a")
    print(f"Elapsed:             {elapsed:.1f}s")
    print(f"Dump written to:     {out_path}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
