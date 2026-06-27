"""Slice H3 verification deliverable: real atomic facts (taken verbatim
from results/decomposition_dump.json, Slice H1's real output) run through
the real LLMFactVerifier against hand-constructed evidence passages, one
example per label (SUPPORTED, CONTRADICTED, NOT_MENTIONED).

Disclosed plainly, per this project's anti-fabrication discipline: the
ATOMIC FACTS are real, unmodified Slice H1 output. The EVIDENCE PASSAGES
are illustrative, hand-written for this demonstration -- the dataset's
character backstories are themselves hackathon-authored summaries, not
verbatim novel text, so there is no guarantee a literal supporting or
contradicting sentence exists in the source novel for any given fact. This
script does not call the retrieval pipeline (lncvs.retrieval is never
imported) -- it stays within Slice H3's stated scope of verifying the
verifier itself, not wiring it into evaluation.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from lncvs.llm import CachingStructuredLLMClient, GeminiStructuredClient, JsonlStructuredLLMCache, LLMConfig  # noqa: E402
from lncvs.reasoning.fact_verification import FactVerificationConfig, LLMFactVerifier  # noqa: E402
from lncvs.reasoning.fact_verification.llm_prompts import SYSTEM_PROMPT  # noqa: E402
from lncvs.schemas import AtomicClaim, FusedEvidence, RetrievalSource  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
VERIFICATION_MODEL = "gemini-2.5-flash"

EXAMPLES = [
    {
        "label_under_test": "SUPPORTED",
        "fact_text": "Faria was shipped to the Chateau d'If for life.",  # real, from decomposition_dump.json row 137
        "evidence_text": (
            "The judges, having weighed the renewed accusations against the old abbe, ordered that Faria "
            "be shipped to the Chateau d'If, there to remain a prisoner for the rest of his life."
        ),
    },
    {
        "label_under_test": "CONTRADICTED",
        "fact_text": "Thalcave's mother died giving birth.",  # real, from decomposition_dump.json row 46
        "evidence_text": (
            "Thalcave's mother survived the difficult birth and lived for many more years, teaching her son "
            "the old songs of the pampas before she finally passed in her old age."
        ),
    },
    {
        "label_under_test": "NOT_MENTIONED",
        "fact_text": "Thalcave learned to tame horses during his boyhood.",  # real, from decomposition_dump.json row 46
        "evidence_text": (
            "The expedition pressed on through the rain-soaked valley, their supplies dwindling, while "
            "Glenarvan studied the weathered map by lantern light for any sign of the missing captain."
        ),
    },
]


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    llm_config = LLMConfig(model_name=VERIFICATION_MODEL, temperature=0.0, max_tokens=2048)
    fact_verification_config = FactVerificationConfig()
    real_client = GeminiStructuredClient(config=llm_config, system_prompt=SYSTEM_PROMPT)
    cache = JsonlStructuredLLMCache(RESULTS_DIR / "fact_verification_cache.jsonl")
    caching_client = CachingStructuredLLMClient(real_client, cache, llm_config, fact_verification_config.schema_version)
    verifier = LLMFactVerifier(caching_client, fact_verification_config)

    results = []
    print("\n" + "=" * 70)
    print("SLICE H3 -- LLMFactVerifier REAL-EXAMPLE DEMONSTRATION")
    print("=" * 70)
    for example in EXAMPLES:
        claim = AtomicClaim(claim_id=f"demo-{example['label_under_test'].lower()}", text=example["fact_text"])
        evidence = FusedEvidence(
            atomic_claim_id=claim.claim_id,
            chunk_id=f"demo-chunk-{example['label_under_test'].lower()}",
            text=example["evidence_text"],
            rrf_score=1.0,
            contributing_sources=[RetrievalSource.SEMANTIC],
            contributing_query_ids=["demo-query"],
        )

        verification = verifier.verify(claim, [evidence])[0]

        print(f"\n--- Expected label under test: {example['label_under_test']} ---")
        print(f"ATOMIC FACT:\n  {example['fact_text']}")
        print(f"RETRIEVED EVIDENCE:\n  {example['evidence_text']}")
        print("FACT VERIFICATION:")
        print(f"  label:             {verification.label.value}")
        print(f"  confidence:        {verification.confidence:.3f}")
        print(f"  supporting_quotes: {verification.supporting_quotes}")
        print(f"  explanation:       {verification.explanation}")

        results.append({
            "label_under_test": example["label_under_test"],
            "fact_text": example["fact_text"],
            "evidence_text": example["evidence_text"],
            "verification": {
                "label": verification.label.value,
                "confidence": verification.confidence,
                "supporting_quotes": list(verification.supporting_quotes),
                "explanation": verification.explanation,
            },
        })

    out_path = RESULTS_DIR / "fact_verification_demo.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDemo output written to: {out_path}")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
