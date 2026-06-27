"""Long-Narrative Validation: proves the real LangGraph pipeline executes
correctly on a genuinely long narrative.

Narrative: data/In search of the castaways.txt (~139k words, ~826k chars).
Chunking: chunk_size=700, overlap=120 -- empirically derived (see
derive_chunk_size_stats() below) against the real cross-encoder NLI
tokenizer: the largest chunk size whose worst-case (premise, hypothesis)
pair-token count stays comfortably under NLILabel max_length=256.

Tier 0: the first gold claim alone -- proves the real pipeline executes at
scale (no exception, no OOM, emits a FinalVerdict).
Tier 1: all 8 gold claims (4 CONSISTENT, 2 CONTRADICTORY, 2
INSUFFICIENT_EVIDENCE) -- the meaningful smoke test, with JSON + Markdown
reports.

Zero production-code changes beyond CrossEncoderNLIModel.tokenizer (a
read-only accessor added for this script's truncation-auditing). This
script drives the REAL, unmodified lncvs.orchestration.graph.build_graph()
via .stream(stream_mode="updates") -- not PipelineRunner, not a
reimplementation -- to obtain per-node latency with zero changes to
orchestration/nodes.py or orchestration/graph.py. invoke() and
stream(stream_mode="updates") execute the identical node sequence and
produce the identical final state (verified empirically during design),
so Phase 7's equivalence guarantee is untouched.

FakeLLMClient (tests/llm/fakes.py) is used for claim decomposition and
question generation because no real LLM provider client exists anywhere
in this codebase yet (see lncvs/llm/__init__.py) -- this is the same
pattern every gated real-model acceptance test in tests/acceptance/ already
uses; it is not specific to this script.
"""

import argparse
import json
import logging
import random
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from lncvs.chunking import ChunkingConfig, chunk_document  # noqa: E402
from lncvs.evaluation import load_dataset, ledger_fingerprint  # noqa: E402
from lncvs.evaluation.dataset import map_spans_to_chunks  # noqa: E402
from lncvs.evaluation.metrics.citation import compute_citation_metrics  # noqa: E402
from lncvs.indexing import (  # noqa: E402
    BM25Index,
    CachingEmbedder,
    ChromaIndex,
    EmbeddingConfig,
    InMemoryEmbeddingCache,
    SentenceTransformerEmbedder,
)
from lncvs.ingestion import load_and_clean_narrative  # noqa: E402
from lncvs.llm import LLMConfig  # noqa: E402
from lncvs.orchestration import PipelineResources, RunContext, build_graph  # noqa: E402
from lncvs.orchestration.state_channels import GraphChannels  # noqa: E402
from lncvs.reasoning.decomposition import DecompositionConfig  # noqa: E402
from lncvs.reasoning.decomposition.prompts import render_decomposition_prompt  # noqa: E402
from lncvs.reasoning.nli import CachingNLIModel, CrossEncoderNLIModel, InMemoryNLICache, NLIConfig  # noqa: E402
from lncvs.reasoning.questions import QuestionGenerationConfig  # noqa: E402
from lncvs.rules import RuleEngineConfig  # noqa: E402
from lncvs.schemas import AblationVariant, ControlState, EvidenceLedger, PipelineStage, VerdictEnum  # noqa: E402
from tests.llm.fakes import FakeLLMClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("validate_long_narrative")

NARRATIVE_PATH = REPO_ROOT / "data" / "In search of the castaways.txt"
GOLD_DATASET_PATH = REPO_ROOT / "datasets" / "castaways_smoke_claims.jsonl"
RESULTS_DIR = REPO_ROOT / "results"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
NLI_MAX_LENGTH = 256
CANDIDATE_CHUNK_SIZES = [600, 700, 800]
SAMPLE_COUNT = 30
WORST_CASE_HYPOTHESIS = (
    "Lord Glenarvan and his companions aboard the Duncan discovered a message in a "
    "bottle that led them on a long search across multiple continents."
)

DECOMPOSITION_BY_CLAIM: dict[str, list[str]] = {
    "The yacht Duncan belonged to Lord Glenarvan.": ["The yacht Duncan belonged to Lord Glenarvan."],
    "Lady Helena was Lord Glenarvan's wife.": ["Lady Helena was Lord Glenarvan's wife."],
    "Mary Grant and Robert were the children of Captain Grant.": [
        "Mary Grant was a child of Captain Grant.",
        "Robert was a child of Captain Grant.",
    ],
    "Jacques Paganel was secretary of the Geographical Society of Paris.": [
        "Jacques Paganel was secretary of the Geographical Society of Paris."
    ],
    "The yacht Duncan belonged to Major MacNabb.": ["The yacht Duncan belonged to Major MacNabb."],
    "Mary Grant and Robert were the children of Lord Glenarvan.": [
        "Mary Grant was a child of Lord Glenarvan.",
        "Robert was a child of Lord Glenarvan.",
    ],
    "Lord Glenarvan communicated with the Duncan using a satellite phone.": [
        "Lord Glenarvan communicated with the Duncan using a satellite phone."
    ],
    "Jacques Paganel sent an email to the Geographical Society of Paris.": [
        "Jacques Paganel sent an email to the Geographical Society of Paris."
    ],
}


def _peak_rss_mb() -> float:
    """Peak resident set size in MB. ru_maxrss is bytes on macOS/darwin, KB on Linux."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def derive_chunk_size_stats(text: str, tokenizer) -> dict[int, dict[str, int]]:
    """Sample real passages at each candidate chunk size against the real NLI
    tokenizer (premise=passage, hypothesis=a deliberately long worst-case
    claim) and report pair-token statistics. Used to empirically validate
    -- not dynamically choose -- the mandated CHUNK_SIZE=700."""
    rng = random.Random(0)
    stats_by_size: dict[int, dict[str, int]] = {}
    for size in CANDIDATE_CHUNK_SIZES:
        samples = []
        for _ in range(SAMPLE_COUNT):
            start = rng.randint(0, len(text) - size - 1)
            passage = text[start : start + size]
            encoded = tokenizer(passage, WORST_CASE_HYPOTHESIS, truncation=False)
            samples.append(len(encoded["input_ids"]))
        samples.sort()
        stats_by_size[size] = {
            "min": samples[0],
            "median": samples[len(samples) // 2],
            "p95": samples[int(0.95 * len(samples))],
            "max": samples[-1],
        }
    return stats_by_size


def token_count_distribution(chunks, tokenizer) -> dict[str, float]:
    """Real per-chunk (premise-only) token-count distribution over the actual chunk corpus."""
    counts = [len(tokenizer(chunk.text, truncation=False)["input_ids"]) for chunk in chunks]
    counts.sort()
    return {
        "min": counts[0],
        "median": statistics.median(counts),
        "p95": counts[int(0.95 * len(counts))],
        "max": counts[-1],
        "mean": statistics.mean(counts),
    }


def count_nli_truncations(nli_results, tokenizer) -> int:
    """Count (premise, hypothesis) pairs from a run's nli_results whose true,
    untruncated token count exceeds NLI_MAX_LENGTH -- these would have been
    silently truncated by the real CrossEncoder during predict()."""
    truncated = 0
    for result in nli_results:
        encoded = tokenizer(result.premise, result.hypothesis, truncation=False)
        if len(encoded["input_ids"]) > NLI_MAX_LENGTH:
            truncated += 1
    return truncated


def build_decomposition_llm() -> FakeLLMClient:
    scripts = {
        render_decomposition_prompt(claim): json.dumps(atomics)
        for claim, atomics in DECOMPOSITION_BY_CLAIM.items()
    }
    return FakeLLMClient(scripted=scripts)


def run_claim_through_graph(
    resources: PipelineResources, narrative_path: Path, original_claim: str
) -> tuple[EvidenceLedger, ControlState, dict[str, float]]:
    """Drive the REAL, unmodified orchestration.graph.build_graph() via
    .stream(stream_mode="updates") to obtain per-node latency without
    touching any node logic. Mirrors only LangGraphPipeline.run()'s ~8-line
    setup (initial state + configurable dict) -- the compiled graph, its 7
    node functions, and its edges are 100% reused unmodified."""
    compiled = build_graph()
    run_context = RunContext(variant=AblationVariant(name="full"))
    initial_state = GraphChannels(
        ledger=EvidenceLedger(original_claim=original_claim),
        control=ControlState(current_stage=PipelineStage.INGESTION, config_fingerprint="long-narrative-validation"),
    )
    config = {
        "configurable": {
            "resources": resources,
            "run_context": run_context,
            "narrative_path": narrative_path,
        }
    }

    node_latencies: dict[str, float] = {}
    final_ledger = initial_state.ledger
    final_control = initial_state.control
    t_prev = time.perf_counter()
    for update in compiled.stream(initial_state, config=config, stream_mode="updates"):
        t_now = time.perf_counter()
        node_name = next(iter(update))
        node_latencies[node_name] = t_now - t_prev
        t_prev = t_now
        node_output = update[node_name]
        if "ledger" in node_output:
            final_ledger = node_output["ledger"]
        if "control" in node_output:
            final_control = node_output["control"]

    return final_ledger, final_control, node_latencies


def evaluate_gold_sanity(expected: VerdictEnum, actual: VerdictEnum) -> tuple[bool, str]:
    """Gold-sanity rule, per the approved plan:
    - CONTRADICTORY/INSUFFICIENT_EVIDENCE gold must match exactly.
    - CONSISTENT gold may resolve to CONSISTENT or INSUFFICIENT_EVIDENCE
      (a recorded retrieval/NLI sensitivity finding, never a failure) but
      must NEVER resolve to CONTRADICTORY.
    """
    if expected is VerdictEnum.CONSISTENT:
        if actual is VerdictEnum.CONTRADICTORY:
            return False, "CONSISTENT claim resolved to CONTRADICTORY -- a real failure"
        if actual is VerdictEnum.INSUFFICIENT_EVIDENCE:
            return True, "CONSISTENT claim resolved to INSUFFICIENT_EVIDENCE -- sensitivity finding, not a failure"
        return True, "matched"
    return (actual == expected), "matched" if actual == expected else f"expected {expected.value}, got {actual.value}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Long-Narrative Validation")
    parser.add_argument("--tier", choices=["0", "1"], default="1", help="0 = single claim, 1 = full 8-claim set")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"started_at": datetime.now(timezone.utc).isoformat()}

    logger.info("Loading real models...")
    t0 = time.perf_counter()
    real_embedder = SentenceTransformerEmbedder(EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    embedding_load_time = time.perf_counter() - t0
    cached_embedder = CachingEmbedder(
        real_embedder, InMemoryEmbeddingCache(), EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    t0 = time.perf_counter()
    real_nli_model = CrossEncoderNLIModel(NLIConfig(model_name="cross-encoder/nli-deberta-v3-base"))
    nli_load_time = time.perf_counter() - t0
    nli_config = NLIConfig(model_name="cross-encoder/nli-deberta-v3-base")
    cached_nli_model = CachingNLIModel(real_nli_model, InMemoryNLICache(), nli_config)
    tokenizer = real_nli_model.tokenizer

    report["model_load_time_seconds"] = {"embedder": embedding_load_time, "nli": nli_load_time}
    logger.info("Models loaded: embedder=%.1fs nli=%.1fs", embedding_load_time, nli_load_time)

    logger.info("Loading narrative: %s", NARRATIVE_PATH)
    document = load_and_clean_narrative(NARRATIVE_PATH, source_id=str(NARRATIVE_PATH))

    logger.info("Validating chunk_size=%d against the real NLI tokenizer...", CHUNK_SIZE)
    chunk_size_stats = derive_chunk_size_stats(document.cleaned_text, tokenizer)
    report["chunk_size_derivation"] = {
        "candidates_sampled": chunk_size_stats,
        "chosen_chunk_size": CHUNK_SIZE,
        "chosen_overlap": CHUNK_OVERLAP,
    }
    chosen_max = chunk_size_stats[CHUNK_SIZE]["max"]
    if chosen_max > NLI_MAX_LENGTH:
        logger.warning(
            "chunk_size=%d's sampled max pair-tokens (%d) EXCEEDS max_length=%d on this run -- "
            "truncation is possible; see nli_truncation_count in the report.",
            CHUNK_SIZE, chosen_max, NLI_MAX_LENGTH,
        )
    else:
        logger.info(
            "chunk_size=%d confirmed safe: sampled max pair-tokens=%d, margin=%d under max_length=%d",
            CHUNK_SIZE, chosen_max, NLI_MAX_LENGTH - chosen_max, NLI_MAX_LENGTH,
        )

    chunking_config = ChunkingConfig(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    chunks = chunk_document(document, chunking_config)
    report["chunk_count"] = len(chunks)
    report["token_count_distribution"] = token_count_distribution(chunks, tokenizer)
    logger.info("Chunked narrative into %d chunks", len(chunks))

    logger.info("Calibration pass: building Chroma + BM25 indices to isolate index-build time...")
    t0 = time.perf_counter()
    calibration_chroma = ChromaIndex(embedder=cached_embedder, collection_name="long-narrative-calibration")
    calibration_chroma.index(chunks)
    chroma_build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    calibration_bm25 = BM25Index(collection_name="long-narrative-calibration-bm25")
    calibration_bm25.index(chunks)
    bm25_build_time = time.perf_counter() - t0
    report["index_build_time_seconds"] = {"chroma": chroma_build_time, "bm25": bm25_build_time}
    logger.info("Calibration index build: chroma=%.1fs bm25=%.1fs", chroma_build_time, bm25_build_time)

    resources = PipelineResources(
        embedder=cached_embedder,
        nli_model=cached_nli_model,
        decomposition_llm=build_decomposition_llm(),
        question_llm=FakeLLMClient(default_response="[]"),
        decomposition_config=DecompositionConfig(llm_config=LLMConfig(model_name="fake-model")),
        question_config=QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
        rule_config=RuleEngineConfig(contradiction_threshold=0.5, entailment_threshold=0.5),
        chunking_config=chunking_config,
        retrieval_top_k=10,
    )

    dataset = load_dataset(GOLD_DATASET_PATH, dataset_id="castaways-smoke")
    examples = dataset.examples if args.tier == "1" else dataset.examples[:1]
    logger.info("Running Tier %s: %d claim(s)", args.tier, len(examples))

    claim_results = []
    all_nli_truncations = 0
    for example in examples:
        logger.info("Running claim %r: %s", example.example_id, example.original_claim)
        t_total = time.perf_counter()
        ledger, control, node_latencies = run_claim_through_graph(resources, NARRATIVE_PATH, example.original_claim)
        total_latency = time.perf_counter() - t_total

        failed = control.current_stage is PipelineStage.ERROR
        predicted_verdict = ledger.final_verdict.verdict if ledger.final_verdict else None

        gold_chunk_ids = map_spans_to_chunks(
            list(example.gold_evidence) + list(example.gold_contradicting_spans), chunks
        )
        citation_metrics = compute_citation_metrics(ledger, gold_chunk_ids)

        grounded = None
        if example.expected_verdict is VerdictEnum.CONTRADICTORY and example.gold_contradicting_spans:
            expected_chunk_ids = map_spans_to_chunks(example.gold_contradicting_spans, chunks)
            cited_chunk_ids = {c.evidence_chunk_id for c in ledger.contradictions}
            grounded = bool(cited_chunk_ids & expected_chunk_ids)

        truncations = count_nli_truncations(ledger.nli_results, tokenizer)
        all_nli_truncations += truncations

        sanity_ok, sanity_note = (
            (False, f"node failure: {[e.message for e in control.errors]}")
            if failed
            else evaluate_gold_sanity(example.expected_verdict, predicted_verdict)
        )

        claim_results.append(
            {
                "example_id": example.example_id,
                "original_claim": example.original_claim,
                "expected_verdict": example.expected_verdict.value,
                "predicted_verdict": predicted_verdict.value if predicted_verdict else None,
                "sanity_ok": sanity_ok,
                "sanity_note": sanity_note,
                "node_failed": failed,
                "node_latencies_seconds": node_latencies,
                "total_latency_seconds": total_latency,
                "retrieval_query_count": len(ledger.retrieval_queries),
                "nli_pair_count": len(ledger.nli_results),
                "nli_truncation_count": truncations,
                "evidence_grounded": grounded,
                "citation_accuracy": citation_metrics.citation_accuracy if citation_metrics else None,
                "ledger_fingerprint": ledger_fingerprint(ledger),
            }
        )
        logger.info(
            "  -> %s (expected %s) total=%.2fs sanity_ok=%s",
            predicted_verdict.value if predicted_verdict else "ERROR",
            example.expected_verdict.value,
            total_latency,
            sanity_ok,
        )

    logger.info("Determinism check: re-running claim 1...")
    determinism_claim = examples[0]
    ledger_a, _, _ = run_claim_through_graph(resources, NARRATIVE_PATH, determinism_claim.original_claim)
    ledger_b, _, _ = run_claim_through_graph(resources, NARRATIVE_PATH, determinism_claim.original_claim)
    determinism_match = ledger_fingerprint(ledger_a) == ledger_fingerprint(ledger_b)
    report["determinism"] = {
        "claim": determinism_claim.example_id,
        "fingerprint_a": ledger_fingerprint(ledger_a),
        "fingerprint_b": ledger_fingerprint(ledger_b),
        "match": determinism_match,
    }
    logger.info("Determinism match: %s", determinism_match)

    report["peak_rss_mb"] = _peak_rss_mb()
    report["claims"] = claim_results
    report["total_nli_truncations"] = all_nli_truncations
    report["verdict_distribution"] = {
        v.value: sum(1 for c in claim_results if c["predicted_verdict"] == v.value) for v in VerdictEnum
    }
    report["all_three_verdict_classes_observed"] = sum(1 for v in report["verdict_distribution"].values() if v > 0) >= (
        3 if args.tier == "1" else 0
    )
    report["all_sanity_checks_passed"] = all(c["sanity_ok"] for c in claim_results)
    report["any_node_failure"] = any(c["node_failed"] for c in claim_results)
    report["tier"] = args.tier
    report["finished_at"] = datetime.now(timezone.utc).isoformat()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"long_narrative_validation_tier{args.tier}_{timestamp}.json"
    md_path = RESULTS_DIR / f"long_narrative_validation_tier{args.tier}_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    logger.info("Reports written: %s, %s", json_path, md_path)

    print_summary(report)

    success = (not report["any_node_failure"]) and report["all_sanity_checks_passed"] and report["determinism"]["match"]
    return 0 if success else 1


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Long-Narrative Validation Report",
        "",
        f"- Tier: {report['tier']}",
        f"- Started: {report['started_at']}",
        f"- Finished: {report['finished_at']}",
        f"- Chunk count: {report['chunk_count']}",
        f"- Peak RSS: {report['peak_rss_mb']:.1f} MB",
        f"- Model load time: embedder={report['model_load_time_seconds']['embedder']:.1f}s, "
        f"nli={report['model_load_time_seconds']['nli']:.1f}s",
        f"- Index build time: chroma={report['index_build_time_seconds']['chroma']:.1f}s, "
        f"bm25={report['index_build_time_seconds']['bm25']:.1f}s",
        f"- Total NLI truncations: {report['total_nli_truncations']}",
        f"- Determinism match: {report['determinism']['match']}",
        f"- All sanity checks passed: {report['all_sanity_checks_passed']}",
        f"- Any node failure: {report['any_node_failure']}",
        "",
        "## Verdict distribution",
        "",
    ]
    for verdict, count in report["verdict_distribution"].items():
        lines.append(f"- {verdict}: {count}")
    lines.append("")
    lines.append("## Per-claim results")
    lines.append("")
    lines.append("| example_id | expected | predicted | sanity_ok | total_latency_s | grounded | citation_accuracy |")
    lines.append("|---|---|---|---|---|---|---|")
    for claim in report["claims"]:
        lines.append(
            f"| {claim['example_id']} | {claim['expected_verdict']} | {claim['predicted_verdict']} | "
            f"{claim['sanity_ok']} | {claim['total_latency_seconds']:.2f} | {claim['evidence_grounded']} | "
            f"{claim['citation_accuracy']} |"
        )
    return "\n".join(lines) + "\n"


def print_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("LONG-NARRATIVE VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Tier: {report['tier']}")
    print(f"Chunk count: {report['chunk_count']}")
    print(f"Token distribution: {report['token_count_distribution']}")
    print(f"Peak RSS: {report['peak_rss_mb']:.1f} MB")
    print(f"Model load time: {report['model_load_time_seconds']}")
    print(f"Index build time: {report['index_build_time_seconds']}")
    print(f"Verdict distribution: {report['verdict_distribution']}")
    print(f"Determinism match: {report['determinism']['match']}")
    print(f"All sanity checks passed: {report['all_sanity_checks_passed']}")
    print(f"Any node failure: {report['any_node_failure']}")
    print(f"Total NLI truncations: {report['total_nli_truncations']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    sys.exit(main())
