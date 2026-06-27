"""H4.5 validation run (pre-H5 gate): compares CrossEncoderFactVerifier vs
LLMFactVerifier on ~20 representative train.csv rows, holding Atomic
Decomposition, Retrieval, Graph, Fusion, and the Rule Engine IDENTICAL
between the two conditions.

This is NOT the full H5 evaluation. Per the user's explicit instruction,
this script must not tune thresholds, must not modify prompts, and must
not change retrieval -- it measures the verifier architecture exactly as
implemented in evaluate_with_graph.py, on a small sample, before spending
further API credits on the complete dataset.

Design: for each selected row, run_claim() is called TWICE -- once with
each FactVerifier -- but decomposition and retrieval are deterministic
and cache-backed (Phase H1/H4-final), so both calls produce byte-identical
atomic_claims/retrieved_evidence/fused_evidence; only the verification
step (and everything downstream of it: to_nli_results -> classify ->
ThresholdRuleEngine) differs. This is the same "identical upstream,
varying only the injected FactVerifier" structure already proven in
tests/scripts/test_evaluate_with_graph_wiring.py, just run against the
real dataset/models instead of fakes.

Token/cost figures are ESTIMATES (char_count // 4), disclosed as such --
no token usage is exposed anywhere in lncvs.llm today (confirmed by
inspection), so an exact accounting is not available without modifying
the frozen LLM client classes, which is out of scope for this validation
run.

Model note (disclosed, not a silent substitution): the LLM verifier in
THIS validation run uses gemini-2.5-flash-lite, not gemini-2.5-flash
(evaluate_with_graph.py's production VERIFIER_MODE="llm" default). The
real gemini-2.5-flash key hit a hard free-tier cap of 20
generate_content requests/day mid-run (confirmed via the API's own
QuotaFailure response, quotaId GenerateRequestsPerDayPerProjectPerModel-
FreeTier, quotaValue 20) -- a daily cap that retry/backoff cannot work
around. flash-lite is a separate model with a separate, higher free-tier
allowance, used here ONLY to let this 20-row comparison complete today.
This is an LLMFactVerifier built locally in this script (NOT via
evaluate_with_graph.build_fact_verifier, whose VERIFICATION_MODEL
constant remains gemini-2.5-flash and is NOT modified by this change) --
evaluate_with_graph.py is untouched, so the eventual full H5 run is free
to use either model, decided separately. Decomposition stays on
gemini-2.5-flash because every row in this sample already has a real,
cached decomposition from Phase H1 (results/decomposition_cache.jsonl) --
reusing it costs zero new API calls regardless of which model the
verifier uses.
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
    DEFAULT_ENTAILMENT_THRESHOLD,
    GOLD_LABEL_TO_VERDICT,
    DatasetRow,
    build_claim_text,
    load_csv,
)
from evaluate_with_graph import build_fact_verifier, build_novel_graph, run_claim  # noqa: E402

from lncvs.evaluation.metrics.verdict import compute_verdict_metrics  # noqa: E402
from lncvs.indexing import CachingEmbedder, EmbeddingConfig, InMemoryEmbeddingCache, SentenceTransformerEmbedder  # noqa: E402
from lncvs.llm import (  # noqa: E402
    CachingLLMClient,
    CachingStructuredLLMClient,
    GeminiLLMClient,
    GeminiStructuredClient,
    JsonlLLMCache,
    JsonlStructuredLLMCache,
    LLMConfig,
    OpenAIStructuredClient,
)
from lncvs.reasoning.decomposition import DecompositionConfig  # noqa: E402
from lncvs.reasoning.fact_verification import FactVerificationConfig, LLMFactVerifier  # noqa: E402
from lncvs.reasoning.fact_verification.llm_prompts import SYSTEM_PROMPT as FACT_VERIFICATION_SYSTEM_PROMPT  # noqa: E402
from lncvs.reasoning.nli import CachingNLIModel, CrossEncoderNLIModel, InMemoryNLICache, NLIConfig  # noqa: E402
from lncvs.rules import RuleEngineConfig  # noqa: E402
from lncvs.schemas import VerdictEnum  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("validate_h5_sample")

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
DECOMPOSITION_MODEL = "gemini-2.5-flash"
# See module docstring "Model note": both gemini-2.5-flash AND
# gemini-2.5-flash-lite hit this free-tier project's ~20-requests/day cap
# (confirmed via two separate QuotaFailure responses, each model has its
# own 20/day pool but the project itself has no further headroom today).
# A real OpenAI key was supplied next, and OpenAIStructuredClient already
# exists in this codebase (used, frozen, for G2 graph extraction's real
# gpt-4o-2024-08-06 calls) -- VERIFIER_PROVIDER selects which one this
# validation's LLMFactVerifier uses, again touching ONLY this script, never
# evaluate_with_graph.py's own VERIFICATION_MODEL/build_fact_verifier.
VERIFIER_PROVIDER = "openai"  # "gemini" or "openai"
VERIFICATION_MODEL = "gpt-4o-2024-08-06" if VERIFIER_PROVIDER == "openai" else "gemini-2.5-flash-lite"
SAMPLE_SIZE = 20

# Call-reduction strategy (added after the free-tier project hit its
# ~20-requests/day cap on BOTH gemini-2.5-flash and gemini-2.5-flash-lite
# mid-run, confirmed via the API's own QuotaFailure response on each):
# CrossEncoderFactVerifier costs ZERO API calls (a local model), so it is
# run across the FULL 20-row sample unconditionally. The scarce
# LLMFactVerifier budget is then spent only on a small, deliberately
# stratified subset of those 20 rows -- prioritizing rows where
# CrossEncoder got the gold label WRONG (the only rows where an "LLM
# corrected CrossEncoder" finding is even possible), with a few
# CrossEncoder-correct rows mixed in specifically to check for LLM
# regressions. This is a sampling-efficiency change to THIS validation
# harness only -- it does not touch retrieval, fusion, decomposition, the
# rule engine, or either FactVerifier's prompts/thresholds, and every row
# still gets a real CrossEncoder verdict either way.
# OpenAI (gpt-4o-2024-08-06, billing-enabled) has no comparable daily cap,
# so with VERIFIER_PROVIDER="openai" the budget is set to the FULL sample --
# the original 20-row request is honored exactly. This constant stays in
# place (rather than being deleted) so a future return to a free-tier-only
# Gemini key can re-impose a smaller budget by changing only this number.
LLM_VERIFIER_ROW_BUDGET = SAMPLE_SIZE if VERIFIER_PROVIDER == "openai" else 6
LLM_VERIFIER_WRONG_ROW_QUOTA = LLM_VERIFIER_ROW_BUDGET if VERIFIER_PROVIDER == "openai" else 4


def build_llm_verifier_with_model(model_name: str, provider: str = "gemini") -> LLMFactVerifier:
    """Mirrors evaluate_with_graph.build_fact_verifier's "llm" branch
    exactly, parameterized by model_name AND provider -- lets this
    validation script pick a different model/provider than
    evaluate_with_graph.py's own VERIFICATION_MODEL constant without
    modifying that frozen file. provider="openai" uses
    OpenAIStructuredClient (already used, frozen, by G2 graph extraction
    for real gpt-4o-2024-08-06 calls) instead of GeminiStructuredClient --
    both implement the identical StructuredLLMClient protocol, so
    LLMFactVerifier (and everything downstream of it) is unaware which one
    it was given."""
    llm_config = LLMConfig(model_name=model_name, temperature=0.0, max_tokens=2048)
    fact_verification_config = FactVerificationConfig()
    if provider == "openai":
        real_client = OpenAIStructuredClient(config=llm_config, system_prompt=FACT_VERIFICATION_SYSTEM_PROMPT, schema_name="fact_verification")
    elif provider == "gemini":
        real_client = GeminiStructuredClient(config=llm_config, system_prompt=FACT_VERIFICATION_SYSTEM_PROMPT)
    else:
        raise ValueError(f"Unknown provider {provider!r}; expected 'gemini' or 'openai'")
    safe_model_name = model_name.replace(".", "_").replace("-", "_")
    cache = JsonlStructuredLLMCache(RESULTS_DIR / f"fact_verification_cache_{provider}_{safe_model_name}.jsonl")
    caching_client = CachingStructuredLLMClient(real_client, cache, llm_config, fact_verification_config.schema_version)
    return LLMFactVerifier(caching_client, fact_verification_config)

# Published Gemini 2.5 Flash per-token rates (text, non-thinking output),
# as of this writing -- used ONLY to produce an order-of-magnitude cost
# estimate alongside the char//4 token-count proxy. Not a precise billing
# reconciliation; no real usage_metadata is captured anywhere in this
# codebase today (confirmed by inspection of lncvs.llm before writing this
# script), so exact-token accounting is out of scope for this validation.
GEMINI_FLASH_INPUT_USD_PER_1M_TOKENS = 0.30
GEMINI_FLASH_OUTPUT_USD_PER_1M_TOKENS = 2.50
ESTIMATED_OUTPUT_TOKENS_PER_CALL = 150  # short structured JSON verdict


def _estimate_tokens(text: str) -> int:
    """char_count // 4, the standard rough English-text proxy. Disclosed
    as an estimate, not a measurement -- see module docstring."""
    return max(1, len(text) // 4)


def select_sample_rows(rows: list[DatasetRow], n: int) -> list[DatasetRow]:
    """Deterministic, label- AND book-proportional sample: train.csv is 51
    consistent / 29 contradict (~64%/36%) across 49 "In Search of the
    Castaways" / 31 "The Count of Monte Cristo" rows (~61%/39%). Within
    each (label, book) group, rows are sorted by NUMERIC row_id ascending
    for reproducibility -- row_id is a numeric string ("1", "10", "100",
    ...), so a plain string sort would badly skew the sample toward
    whichever book happens to have more low-cardinality numeric-string
    ids sort first; this is why int(r.row_id) is used as the sort key.
    No randomness -- the same n on the same CSV always selects the same
    rows."""
    by_group: dict[tuple[str, str], list[DatasetRow]] = {}
    for row in rows:
        by_group.setdefault((row.label, row.book_name), []).append(row)
    for group in by_group.values():
        group.sort(key=lambda r: int(r.row_id))

    total_labeled = len(rows)
    selected: list[DatasetRow] = []
    remaining = n
    groups = sorted(by_group.keys())
    for index, group in enumerate(groups):
        is_last = index == len(groups) - 1
        count = remaining if is_last else round(n * len(by_group[group]) / total_labeled)
        selected.extend(by_group[group][:count])
        remaining -= count
    return selected


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = load_csv(DATA_DIR / "train.csv", has_label=True)
    sample_rows = select_sample_rows(train_rows, SAMPLE_SIZE)
    logger.info("Selected %d rows for H4.5 validation: %s", len(sample_rows), [r.row_id for r in sample_rows])

    logger.info("Loading real models...")
    real_embedder = SentenceTransformerEmbedder(EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    cached_embedder = CachingEmbedder(
        real_embedder, InMemoryEmbeddingCache(), EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    real_nli = CrossEncoderNLIModel(NLIConfig(model_name="cross-encoder/nli-deberta-v3-base"))
    cached_nli = CachingNLIModel(real_nli, InMemoryNLICache(), NLIConfig(model_name="cross-encoder/nli-deberta-v3-base"))

    cross_encoder_verifier = build_fact_verifier("cross_encoder", cached_nli)
    llm_verifier = build_llm_verifier_with_model(VERIFICATION_MODEL, VERIFIER_PROVIDER)

    decomposition_llm_config = LLMConfig(model_name=DECOMPOSITION_MODEL, temperature=0.0, max_tokens=2048)
    decomp_config = DecompositionConfig(llm_config=decomposition_llm_config, max_atomic_claims=10)
    # No GEMINI_API_KEY is configured in this environment (only OpenAI/HF
    # keys are present today). GeminiLLMClient's constructor requires SOME
    # non-empty string to build its genai.Client, but never actually issues
    # a request: every one of this sample's 20 rows was confirmed, by
    # direct cache-key computation against results/decomposition_cache.jsonl
    # before this run, to be a guaranteed cache hit under this exact
    # decomposition_llm_config fingerprint. If that ever stopped being true
    # (e.g. a new row needing a fresh decomposition), this placeholder
    # would surface as a loud Gemini auth error -- never a silent fallback.
    real_decomposition_client = GeminiLLMClient(config=decomposition_llm_config, api_key="unused-cache-only-placeholder")
    decomposition_cache = JsonlLLMCache(RESULTS_DIR / "decomposition_cache.jsonl")
    decomposition_llm = CachingLLMClient(real_decomposition_client, decomposition_cache, decomposition_llm_config)

    # Diagnostic re-run (explicitly approved): consistency_requires_entailment
    # flipped True->this is NOT a threshold tuning change -- it's the strict
    # vs lenient RULE POLICY identified as the H4.5 root cause. The H4.5
    # finding: evaluate_with_graph.py's False setting was empirically tuned
    # around CrossEncoder's known weakness (rarely emits ENTAILMENT) and
    # silently collapses INSUFFICIENT_EVIDENCE into CONSISTENT for ANY
    # unresolved claim -- masked when CrossEncoder fires noisily, fully
    # exposed once a well-calibrated LLM verifier stops firing on absent
    # (not contradicting) evidence. True restores this project's own
    # stated three-verdict semantics (PROJECT_SPEC.md / CLAUDE.md: absence
    # of evidence routes to INSUFFICIENT_EVIDENCE, never CONSISTENT).
    RULE_CONSISTENCY_REQUIRES_ENTAILMENT = True
    rule_config = RuleEngineConfig(
        contradiction_threshold=0.9, entailment_threshold=DEFAULT_ENTAILMENT_THRESHOLD,
        consistency_requires_entailment=RULE_CONSISTENCY_REQUIRES_ENTAILMENT,
    )

    book_resources: dict[str, tuple] = {}
    needed_books = {row.book_name for row in sample_rows}
    for book_name in needed_books:
        book_path = BOOK_NAME_TO_PATH[book_name]
        chunks, chroma_index, bm25_index, graph_retriever, constructed = build_novel_graph(book_name, book_path, cached_embedder)
        book_resources[book_name] = (chunks, chroma_index, bm25_index, graph_retriever, constructed)
        logger.info("Graph ready for %r: %d entities, %d relations, %d events", book_name, len(constructed.entities), len(constructed.relations), len(constructed.events))

    rows_report: list[dict] = []
    cross_pairs: list[tuple[VerdictEnum, VerdictEnum]] = []
    llm_pairs: list[tuple[VerdictEnum, VerdictEnum]] = []
    llm_calls_total = 0
    llm_input_chars_total = 0
    llm_latencies: list[float] = []
    cross_latencies: list[float] = []

    # PASS 1: CrossEncoderFactVerifier over the FULL sample. Zero API calls
    # (a local model) -- every row gets a real verdict regardless of the
    # Gemini quota situation.
    ce_results: dict[str, dict] = {}
    for row in sample_rows:
        claim_text = build_claim_text(row)
        _, chroma_index, bm25_index, graph_retriever, _ = book_resources[row.book_name]
        gold = GOLD_LABEL_TO_VERDICT[row.label]

        t0 = time.perf_counter()
        try:
            ledger_ce, _ = run_claim(claim_text, chroma_index, bm25_index, graph_retriever, decomposition_llm, decomp_config, cross_encoder_verifier, rule_config)
            ce_error = None
        except Exception as exc:
            logger.exception("Row %s: cross_encoder verifier raised", row.row_id)
            ledger_ce = None
            ce_error = str(exc)
        ce_latency = time.perf_counter() - t0
        cross_latencies.append(ce_latency)

        predicted_ce = ledger_ce.final_verdict.verdict if (ledger_ce is not None and ledger_ce.final_verdict) else None
        if predicted_ce is not None:
            cross_pairs.append((gold, predicted_ce))
        ce_results[row.row_id] = {
            "predicted_ce": predicted_ce,
            "ce_error": ce_error,
            "ce_latency": ce_latency,
            "gold": gold,
            "ce_correct": predicted_ce == gold,
        }
        logger.info("Row %s [%s] CROSS_ENCODER: gold=%s predicted=%s (err=%s) correct=%s", row.row_id, row.book_name, gold.value, predicted_ce.value if predicted_ce else None, ce_error, predicted_ce == gold)

    # PASS 2 selection: spend the scarce LLM-verifier budget on the rows
    # most likely to be informative -- CE-wrong rows first (up to
    # LLM_VERIFIER_WRONG_ROW_QUOTA), then fill the remaining budget with
    # CE-correct rows (regression check). Deterministic: both groups
    # preserve select_sample_rows' original (label, book, row_id) order.
    wrong_rows = [row for row in sample_rows if not ce_results[row.row_id]["ce_correct"]]
    correct_rows = [row for row in sample_rows if ce_results[row.row_id]["ce_correct"]]
    llm_eval_rows = wrong_rows[:LLM_VERIFIER_WRONG_ROW_QUOTA]
    remaining_budget = LLM_VERIFIER_ROW_BUDGET - len(llm_eval_rows)
    llm_eval_rows += correct_rows[: max(0, remaining_budget)]
    llm_eval_row_ids = {row.row_id for row in llm_eval_rows}
    logger.info(
        "LLM verifier budget: %d/%d rows selected (%d CE-wrong, %d CE-correct) out of %d total",
        len(llm_eval_rows), LLM_VERIFIER_ROW_BUDGET, sum(1 for r in llm_eval_rows if r.row_id in {x.row_id for x in wrong_rows}), sum(1 for r in llm_eval_rows if r.row_id in {x.row_id for x in correct_rows}), len(sample_rows),
    )

    # PASS 2: LLMFactVerifier on the selected subset ONLY.
    for row in sample_rows:
        ce = ce_results[row.row_id]
        gold = ce["gold"]
        predicted_ce = ce["predicted_ce"]

        if row.row_id not in llm_eval_row_ids:
            rows_report.append(
                {
                    "row_id": row.row_id, "book_name": row.book_name, "char": row.char, "gold": gold.value,
                    "cross_encoder_predicted": predicted_ce.value if predicted_ce else None,
                    "llm_predicted": None, "cross_encoder_error": ce["ce_error"],
                    "llm_error": "skipped: outside today's LLM-verifier call budget (see LLM_VERIFIER_ROW_BUDGET)",
                    "verdict_changed": False, "cross_encoder_correct": ce["ce_correct"], "llm_correct": None,
                    "cross_encoder_latency_s": ce["ce_latency"], "llm_latency_s": None,
                    "n_atomic_claims": None, "n_llm_verifier_calls": 0, "llm_evaluated": False,
                }
            )
            continue

        claim_text = build_claim_text(row)
        _, chroma_index, bm25_index, graph_retriever, _ = book_resources[row.book_name]

        t0 = time.perf_counter()
        try:
            ledger_llm, _ = run_claim(claim_text, chroma_index, bm25_index, graph_retriever, decomposition_llm, decomp_config, llm_verifier, rule_config)
            llm_error = None
        except Exception as exc:
            logger.exception("Row %s: llm verifier raised", row.row_id)
            ledger_llm = None
            llm_error = str(exc)
        llm_latency = time.perf_counter() - t0
        llm_latencies.append(llm_latency)

        predicted_llm = ledger_llm.final_verdict.verdict if (ledger_llm is not None and ledger_llm.final_verdict) else None
        if predicted_llm is not None:
            llm_pairs.append((gold, predicted_llm))

        if ledger_llm is not None:
            n_claims_with_evidence = sum(1 for c in ledger_llm.atomic_claims if any(f.atomic_claim_id == c.claim_id for f in ledger_llm.fused_evidence))
            llm_calls_total += n_claims_with_evidence
            for claim in ledger_llm.atomic_claims:
                evidence_texts = [f.text for f in ledger_llm.fused_evidence if f.atomic_claim_id == claim.claim_id]
                if evidence_texts:
                    llm_input_chars_total += len(claim.text) + sum(len(t) for t in evidence_texts)
        else:
            n_claims_with_evidence = 0

        rows_report.append(
            {
                "row_id": row.row_id,
                "book_name": row.book_name,
                "char": row.char,
                "gold": gold.value,
                "cross_encoder_predicted": predicted_ce.value if predicted_ce else None,
                "llm_predicted": predicted_llm.value if predicted_llm else None,
                "cross_encoder_error": ce["ce_error"],
                "llm_error": llm_error,
                "verdict_changed": predicted_ce != predicted_llm,
                "cross_encoder_correct": ce["ce_correct"],
                "llm_correct": predicted_llm == gold,
                "cross_encoder_latency_s": ce["ce_latency"],
                "llm_latency_s": llm_latency,
                "n_atomic_claims": len(ledger_llm.atomic_claims) if ledger_llm is not None else None,
                "n_llm_verifier_calls": n_claims_with_evidence,
                "llm_evaluated": True,
            }
        )
        logger.info(
            "Row %s [%s]: gold=%s cross_encoder=%s (err=%s) llm=%s (err=%s) changed=%s",
            row.row_id, row.book_name, gold.value, predicted_ce.value if predicted_ce else None, ce["ce_error"],
            predicted_llm.value if predicted_llm else None, llm_error, predicted_ce != predicted_llm,
        )

    cross_metrics = compute_verdict_metrics(cross_pairs) if cross_pairs else None
    llm_metrics = compute_verdict_metrics(llm_pairs) if llm_pairs else None

    llm_evaluated_rows = [r for r in rows_report if r["llm_evaluated"]]
    corrections = [r for r in llm_evaluated_rows if not r["cross_encoder_correct"] and r["llm_correct"]]
    regressions = [r for r in llm_evaluated_rows if r["cross_encoder_correct"] and not r["llm_correct"]]
    verdict_changes = [r for r in llm_evaluated_rows if r["verdict_changed"]]

    avg_llm_calls_per_row = llm_calls_total / len(llm_evaluated_rows) if llm_evaluated_rows else 0.0
    avg_input_tokens_per_call = _estimate_tokens(" " * (llm_input_chars_total // max(1, llm_calls_total))) if llm_calls_total else 0
    avg_output_tokens_per_call = ESTIMATED_OUTPUT_TOKENS_PER_CALL
    cost_per_call = (
        avg_input_tokens_per_call * GEMINI_FLASH_INPUT_USD_PER_1M_TOKENS / 1_000_000
        + avg_output_tokens_per_call * GEMINI_FLASH_OUTPUT_USD_PER_1M_TOKENS / 1_000_000
    )
    estimated_cost_per_row = cost_per_call * avg_llm_calls_per_row

    summary = {
        "sample_size": len(sample_rows),
        "llm_evaluated_sample_size": len(llm_evaluated_rows),
        "row_ids": [r.row_id for r in sample_rows],
        "llm_evaluated_row_ids": sorted(llm_eval_row_ids, key=int),
        "cross_encoder_metrics_full_sample": json.loads(cross_metrics.model_dump_json()) if cross_metrics else None,
        "llm_metrics_subset_only": json.loads(llm_metrics.model_dump_json()) if llm_metrics else None,
        "num_verdict_changes": len(verdict_changes),
        "llm_corrections_of_cross_encoder_mistakes": [
            {"row_id": r["row_id"], "gold": r["gold"], "cross_encoder": r["cross_encoder_predicted"], "llm": r["llm_predicted"]} for r in corrections
        ],
        "llm_regressions_vs_cross_encoder": [
            {"row_id": r["row_id"], "gold": r["gold"], "cross_encoder": r["cross_encoder_predicted"], "llm": r["llm_predicted"]} for r in regressions
        ],
        "avg_cross_encoder_latency_seconds": sum(cross_latencies) / len(cross_latencies) if cross_latencies else 0.0,
        "avg_llm_latency_seconds": sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0.0,
        "avg_llm_verifier_calls_per_row": avg_llm_calls_per_row,
        "total_gemini_verifier_calls": llm_calls_total,
        "estimated_avg_input_tokens_per_llm_call": avg_input_tokens_per_call,
        "estimated_avg_output_tokens_per_llm_call": avg_output_tokens_per_call,
        "estimated_cost_per_row_usd": estimated_cost_per_row,
        "estimated_cost_for_llm_evaluated_subset_usd": estimated_cost_per_row * len(llm_evaluated_rows),
        "rows": rows_report,
    }
    (RESULTS_DIR / "h4_5_validation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"H4.5 VALIDATION: CrossEncoderFactVerifier (n={len(sample_rows)}) vs LLMFactVerifier (n={len(llm_evaluated_rows)} of {len(sample_rows)}, budget-limited)")
    print("=" * 70)
    if cross_metrics:
        print(f"CrossEncoder (full sample, n={len(sample_rows)}): accuracy={cross_metrics.accuracy:.3f} macro_f1={cross_metrics.macro_f1:.3f} macro_p={cross_metrics.macro_precision:.3f} macro_r={cross_metrics.macro_recall:.3f}")
    if llm_metrics:
        print(f"LLM          (subset only, n={len(llm_evaluated_rows)}): accuracy={llm_metrics.accuracy:.3f} macro_f1={llm_metrics.macro_f1:.3f} macro_p={llm_metrics.macro_precision:.3f} macro_r={llm_metrics.macro_recall:.3f}")
        print("NOTE: LLM metrics are computed on the stratified subset ONLY -- not directly comparable to CrossEncoder's full-sample metrics without caveat.")
    print(f"Verdict changes (within subset): {len(verdict_changes)} / {len(llm_evaluated_rows)}")
    print(f"LLM corrections of cross-encoder mistakes: {len(corrections)}")
    print(f"LLM regressions vs cross-encoder: {len(regressions)}")
    print(f"Avg latency: cross_encoder={summary['avg_cross_encoder_latency_seconds']:.2f}s llm={summary['avg_llm_latency_seconds']:.2f}s")
    print(f"Total Gemini verifier calls made: {llm_calls_total}  |  Avg LLM verifier calls/evaluated row: {avg_llm_calls_per_row:.2f}")
    print(f"Estimated cost/row: ${estimated_cost_per_row:.5f}  |  Estimated cost for evaluated subset: ${estimated_cost_per_row * len(llm_evaluated_rows):.4f}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
