"""Evaluation Study: runs the REAL LNCVS pipeline (real LangGraph workflow,
real retrieval, real fusion, real NLI, real rule engine) on the provided
data/train.csv and data/test.csv, with checkpointed predictions and a
threshold sweep over the existing, unmodified ThresholdRuleEngine/classify().

Dataset schema (verified by direct inspection -- see Phase A of the
evaluation report, not assumed):
  train.csv: id, book_name, char, caption, content, label (label in {consistent, contradict})
  test.csv:  id, book_name, char, caption, content  (NO label column -- unlabeled holdout)

No evidence-span/chunk annotations exist in either file. Retrieval metrics
(Recall@k, MRR, gold-grounded citation accuracy) are therefore NOT
computable and are NOT fabricated here -- see the generated report's
"Retrieval Evaluation" section, which states this explicitly rather than
estimating a number.

original_claim is constructed as f"Regarding {char}: {content}" because
many `content` rows refer to the character only by pronoun ("He once
found..."). This resolves the referent deterministically in this script,
the same way scripts/validate_long_narrative.py already constructs claim
strings before handing them to the real pipeline.

Claim decomposition uses FakeLLMClient scripted to pass each constructed
claim through as a single atomic claim, unmodified -- no real LLM provider
client exists anywhere in this codebase (see lncvs/llm/__init__.py), and
every gated real-model test in tests/acceptance/ already relies on this
same fake for decomposition. Retrieval, fusion, NLI, and the rule engine
are 100% real, run via the unmodified lncvs.orchestration.graph.build_graph(),
driven by run_claim_through_graph() imported directly from
validate_long_narrative.py -- no second implementation of the graph driver.
"""

import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from validate_long_narrative import run_claim_through_graph  # noqa: E402

from lncvs.chunking import ChunkingConfig  # noqa: E402
from lncvs.evaluation.metrics.verdict import compute_verdict_metrics  # noqa: E402
from lncvs.indexing import CachingEmbedder, EmbeddingConfig, InMemoryEmbeddingCache, SentenceTransformerEmbedder  # noqa: E402
from lncvs.llm import LLMConfig  # noqa: E402
from lncvs.orchestration import PipelineResources  # noqa: E402
from lncvs.reasoning.decomposition import DecompositionConfig  # noqa: E402
from lncvs.reasoning.decomposition.prompts import render_decomposition_prompt  # noqa: E402
from lncvs.reasoning.nli import CachingNLIModel, CrossEncoderNLIModel, InMemoryNLICache, NLIConfig  # noqa: E402
from lncvs.reasoning.questions import QuestionGenerationConfig  # noqa: E402
from lncvs.rules import RuleEngineConfig, ThresholdRuleEngine  # noqa: E402
from lncvs.schemas import AtomicClaim, EvidenceLedger, NLILabel, NLIResult, PipelineStage, VerdictEnum  # noqa: E402
from tests.llm.fakes import FakeLLMClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("evaluate_dataset")

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

BOOK_NAME_TO_PATH = {
    "In Search of the Castaways": DATA_DIR / "In search of the castaways.txt",
    "The Count of Monte Cristo": DATA_DIR / "The Count of Monte Cristo.txt",
}

GOLD_LABEL_TO_VERDICT = {
    "consistent": VerdictEnum.CONSISTENT,
    "contradict": VerdictEnum.CONTRADICTORY,
}

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
DEFAULT_CONTRADICTION_THRESHOLD = 0.5
DEFAULT_ENTAILMENT_THRESHOLD = 0.5
THRESHOLD_SWEEP = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


@dataclass
class DatasetRow:
    row_id: str
    book_name: str
    char: str
    caption: str
    content: str
    label: str | None  # None for test.csv (no label column exists)


def load_csv(path: Path, has_label: bool) -> list[DatasetRow]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            DatasetRow(
                row_id=r["id"],
                book_name=r["book_name"],
                char=r["char"],
                caption=r.get("caption", ""),
                content=r["content"],
                label=r["label"] if has_label else None,
            )
            for r in reader
        ]
    return rows


def build_claim_text(row: DatasetRow) -> str:
    """Resolve the pronoun-dependent `content` field's referent deterministically."""
    return f"Regarding {row.char}: {row.content}"


def build_decomposition_llm_for_rows(rows: list[DatasetRow]) -> FakeLLMClient:
    scripts = {render_decomposition_prompt(build_claim_text(row)): json.dumps([build_claim_text(row)]) for row in rows}
    return FakeLLMClient(scripted=scripts)


def load_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    checkpoint: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            checkpoint[record["row_id"]] = record
    return checkpoint


def append_checkpoint(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def run_dataset(rows: list[DatasetRow], resources: PipelineResources, checkpoint_path: Path) -> list[dict]:
    """Run every row through the real, unmodified pipeline once, checkpointing
    incrementally so a crash mid-run does not lose progress (resumable: rows
    already present in checkpoint_path are skipped)."""
    checkpoint = load_checkpoint(checkpoint_path)
    results: list[dict] = []
    for row in rows:
        if row.row_id in checkpoint:
            logger.info("Row %s already checkpointed, skipping", row.row_id)
            results.append(checkpoint[row.row_id])
            continue

        narrative_path = BOOK_NAME_TO_PATH.get(row.book_name)
        claim_text = build_claim_text(row)
        if narrative_path is None or not narrative_path.is_file():
            logger.error("Unknown/missing book_name %r for row %s", row.book_name, row.row_id)
            record = {
                "row_id": row.row_id, "book_name": row.book_name, "char": row.char,
                "claim": claim_text, "gold_label": row.label, "error": f"unknown book_name {row.book_name!r}",
                "predicted_verdict": None, "node_failed": True, "atomic_claim_ids": [], "nli_results": [],
                "retrieved_evidence": [], "fused_evidence": [], "node_errors": [], "fired_rule": None,
                "total_latency_seconds": 0.0,
            }
            append_checkpoint(checkpoint_path, record)
            results.append(record)
            continue

        t0 = time.perf_counter()
        try:
            ledger, control, _node_latencies = run_claim_through_graph(resources, narrative_path, claim_text)
            error_text = None
        except Exception as exc:  # the pipeline itself raised -- record, do not crash the whole run
            logger.exception("Row %s raised an exception", row.row_id)
            ledger = None
            control = None
            error_text = str(exc)
        total_latency = time.perf_counter() - t0

        if error_text is not None or control is None:
            record = {
                "row_id": row.row_id, "book_name": row.book_name, "char": row.char,
                "claim": claim_text, "gold_label": row.label, "error": error_text,
                "predicted_verdict": None, "node_failed": True, "atomic_claim_ids": [], "nli_results": [],
                "retrieved_evidence": [], "fused_evidence": [], "node_errors": [], "fired_rule": None,
                "total_latency_seconds": total_latency,
            }
            append_checkpoint(checkpoint_path, record)
            results.append(record)
            continue

        failed = control.current_stage is PipelineStage.ERROR
        predicted_verdict = ledger.final_verdict.verdict.value if (not failed and ledger.final_verdict) else None

        record = {
            "row_id": row.row_id,
            "book_name": row.book_name,
            "char": row.char,
            "claim": claim_text,
            "gold_label": row.label,
            "error": None,
            "predicted_verdict": predicted_verdict,
            "fired_rule": ledger.final_verdict.fired_rule if (not failed and ledger.final_verdict) else None,
            "node_failed": failed,
            "node_errors": [e.message for e in control.errors] if failed else [],
            "total_latency_seconds": total_latency,
            "atomic_claim_ids": [c.claim_id for c in ledger.atomic_claims],
            "retrieved_evidence": [
                {"chunk_id": e.chunk_id, "source": e.source.value, "rank": e.rank, "text": e.text}
                for e in ledger.retrieved_evidence
            ],
            "fused_evidence": [
                {"chunk_id": f.chunk_id, "rrf_score": f.rrf_score, "text": f.text} for f in ledger.fused_evidence
            ],
            "nli_results": [
                {
                    "atomic_claim_id": n.atomic_claim_id,
                    "evidence_chunk_id": n.evidence_chunk_id,
                    "label": n.label.value,
                    "score": n.score,
                    "premise": n.premise,
                    "hypothesis": n.hypothesis,
                }
                for n in ledger.nli_results
            ],
        }
        append_checkpoint(checkpoint_path, record)
        results.append(record)
        logger.info(
            "Row %s [%s / %s]: gold=%s predicted=%s latency=%.2fs",
            row.row_id, row.book_name, row.char, row.label, predicted_verdict, total_latency,
        )

    return results


def to_verdict_pairs(records: list[dict]) -> list[tuple[VerdictEnum, VerdictEnum]]:
    pairs = []
    for r in records:
        if r.get("gold_label") is None or r.get("predicted_verdict") is None:
            continue
        pairs.append((GOLD_LABEL_TO_VERDICT[r["gold_label"]], VerdictEnum(r["predicted_verdict"])))
    return pairs


def reconstruct_ledger_for_sweep(record: dict) -> EvidenceLedger:
    """Rebuild a minimal EvidenceLedger from a checkpointed record's
    atomic_claim_ids + nli_results, so the threshold sweep can call the
    real, unmodified classify()/ThresholdRuleEngine.evaluate() WITHOUT
    re-running retrieval/fusion/NLI for every threshold value."""
    # text content is never read by classify()/ThresholdRuleEngine.evaluate()
    # (only claim_id is used) -- "placeholder" satisfies AtomicClaim's
    # min_length=1 validation without affecting sweep results.
    atomic_claims = [AtomicClaim(claim_id=cid, text="placeholder") for cid in record["atomic_claim_ids"]]
    nli_results = [
        NLIResult(
            atomic_claim_id=n["atomic_claim_id"],
            evidence_chunk_id=n["evidence_chunk_id"],
            label=NLILabel(n["label"]),
            score=n["score"],
            premise=n["premise"],
            hypothesis=n["hypothesis"],
        )
        for n in record["nli_results"]
    ]
    return EvidenceLedger(original_claim=record["claim"], atomic_claims=atomic_claims, nli_results=nli_results)


def run_threshold_sweep(records: list[dict]) -> list[dict]:
    """Sweep contradiction_threshold over THRESHOLD_SWEEP, holding
    entailment_threshold fixed, re-evaluating the REAL ThresholdRuleEngine
    against already-captured NLI results (train.csv only -- the only file
    with gold labels)."""
    sweep_rows = [
        r for r in records if r.get("gold_label") is not None and not r.get("node_failed") and r.get("atomic_claim_ids")
    ]
    results = []
    for threshold in THRESHOLD_SWEEP:
        config = RuleEngineConfig(contradiction_threshold=threshold, entailment_threshold=DEFAULT_ENTAILMENT_THRESHOLD)
        engine = ThresholdRuleEngine(config)
        pairs = []
        for r in sweep_rows:
            ledger = reconstruct_ledger_for_sweep(r)
            verdict = engine.evaluate(ledger)
            pairs.append((GOLD_LABEL_TO_VERDICT[r["gold_label"]], verdict.verdict))
        metrics = compute_verdict_metrics(pairs)
        contradictory = next(m for m in metrics.per_class if m.verdict is VerdictEnum.CONTRADICTORY)
        results.append(
            {
                "contradiction_threshold": threshold,
                "accuracy": metrics.accuracy,
                "precision_contradictory": contradictory.precision,
                "recall_contradictory": contradictory.recall,
                "macro_f1": metrics.macro_f1,
            }
        )
    return results


def classify_failure_mode(record: dict) -> str:
    """Categorize an incorrect prediction's dominant failure layer, supported
    by the actual captured evidence -- never asserted without checking it."""
    gold_verdict = GOLD_LABEL_TO_VERDICT[record["gold_label"]]
    nli_results = record["nli_results"]

    if not nli_results:
        return "retrieval_or_fusion"  # zero evidence ever reached NLI

    target_label = NLILabel.CONTRADICTION.value if gold_verdict is VerdictEnum.CONTRADICTORY else NLILabel.ENTAILMENT.value
    matching = [n for n in nli_results if n["label"] == target_label]

    if not matching:
        return "nli"  # the model never assigned the correct relationship at any confidence

    threshold = DEFAULT_CONTRADICTION_THRESHOLD if gold_verdict is VerdictEnum.CONTRADICTORY else DEFAULT_ENTAILMENT_THRESHOLD
    if all(n["score"] < threshold for n in matching) and record["predicted_verdict"] != gold_verdict.value:
        return "threshold"  # right label exists but below the firing threshold

    if record["predicted_verdict"] not in (gold_verdict.value, VerdictEnum.INSUFFICIENT_EVIDENCE.value):
        return "nli"  # a different label fired with high enough confidence to win (e.g. false-positive contradiction)

    return "unclassified"


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

    t0 = time.perf_counter()
    real_nli = CrossEncoderNLIModel(NLIConfig(model_name="cross-encoder/nli-deberta-v3-base"))
    nli_load_s = time.perf_counter() - t0
    cached_nli = CachingNLIModel(
        real_nli, InMemoryNLICache(), NLIConfig(model_name="cross-encoder/nli-deberta-v3-base")
    )
    logger.info("Models loaded: embedder=%.1fs nli=%.1fs", embedder_load_s, nli_load_s)

    resources = PipelineResources(
        embedder=cached_embedder,
        nli_model=cached_nli,
        decomposition_llm=build_decomposition_llm_for_rows(all_rows),
        question_llm=FakeLLMClient(default_response="[]"),
        decomposition_config=DecompositionConfig(llm_config=LLMConfig(model_name="fake-model")),
        question_config=QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
        rule_config=RuleEngineConfig(
            contradiction_threshold=DEFAULT_CONTRADICTION_THRESHOLD, entailment_threshold=DEFAULT_ENTAILMENT_THRESHOLD
        ),
        chunking_config=ChunkingConfig(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP),
        retrieval_top_k=10,
    )

    logger.info("Running train.csv (%d rows)...", len(train_rows))
    train_results = run_dataset(train_rows, resources, RESULTS_DIR / "predictions_train.jsonl")

    logger.info("Running test.csv (%d rows)...", len(test_rows))
    test_results = run_dataset(test_rows, resources, RESULTS_DIR / "predictions_test.jsonl")

    train_pairs = to_verdict_pairs(train_results)
    verdict_metrics = compute_verdict_metrics(train_pairs) if train_pairs else None

    sweep = run_threshold_sweep(train_results)

    incorrect = [
        r
        for r in train_results
        if r.get("gold_label") in GOLD_LABEL_TO_VERDICT
        and r.get("predicted_verdict") != GOLD_LABEL_TO_VERDICT[r["gold_label"]].value
    ]
    failure_modes: dict[str, list[dict]] = {}
    for r in incorrect:
        mode = classify_failure_mode(r)
        failure_modes.setdefault(mode, []).append(r)

    output = {
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "model_load_seconds": {"embedder": embedder_load_s, "nli": nli_load_s},
        "verdict_metrics": json.loads(verdict_metrics.model_dump_json()) if verdict_metrics else None,
        "threshold_sweep": sweep,
        "incorrect_count": len(incorrect),
        "failure_mode_counts": {mode: len(items) for mode, items in failure_modes.items()},
        "failure_examples": {
            mode: [
                {
                    "row_id": r["row_id"], "book_name": r["book_name"], "char": r["char"], "claim": r["claim"],
                    "gold_label": r["gold_label"], "predicted_verdict": r["predicted_verdict"],
                    "fired_rule": r["fired_rule"], "nli_results": r["nli_results"],
                }
                for r in items
            ]
            for mode, items in failure_modes.items()
        },
    }
    (RESULTS_DIR / "evaluation_analysis.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote results/evaluation_analysis.json")

    print("\n" + "=" * 70)
    print("EVALUATION STUDY SUMMARY")
    print("=" * 70)
    if verdict_metrics:
        print(f"Train accuracy: {verdict_metrics.accuracy:.3f}  macro_f1: {verdict_metrics.macro_f1:.3f}")
    print(f"Incorrect: {len(incorrect)} / {len(train_pairs)}")
    print(f"Failure modes: {output['failure_mode_counts']}")
    print("Threshold sweep:")
    for row in sweep:
        print(f"  t={row['contradiction_threshold']:.2f} acc={row['accuracy']:.3f} macro_f1={row['macro_f1']:.3f}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
