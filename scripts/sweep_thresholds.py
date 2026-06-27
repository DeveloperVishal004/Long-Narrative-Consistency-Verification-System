"""Offline threshold sweep over the persisted NLI dump (no API, no NLI,
no graph rebuild). Recomputes verdicts using the EXACT real rule logic
(lncvs.rules.classification.classify semantics + ThresholdRuleEngine's
Rule 1>2>3 ordering), then reports accuracy / macro-F1 for each
(contradiction_threshold, entailment_threshold) pair.

The verdict recomputation here is a faithful, line-for-line reimplementation
of classify() + ThresholdRuleEngine.evaluate() over the dumped (label,
score) tuples -- it changes no production code, it only evaluates what the
real engine WOULD produce under each threshold pair, so the chosen pair can
then be set once in the real evaluation.
"""

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
NLI_DUMP_PATH = RESULTS_DIR / "threshold_analysis_nli_dump.json"

GOLD = {"consistent": "CONSISTENT", "contradict": "CONTRADICTORY"}
VERDICTS = ["CONSISTENT", "CONTRADICTORY", "INSUFFICIENT_EVIDENCE"]


def verdict_for_row(nli_by_claim, claim_ids, c_thresh, e_thresh):
    """Faithful classify() + ThresholdRuleEngine.evaluate() over dumped scores."""
    statuses = []
    for cid in claim_ids:
        results = nli_by_claim.get(cid, [])
        contradicting = [r for r in results if r["label"] == "CONTRADICTION" and r["score"] >= c_thresh]
        entailing = [r for r in results if r["label"] == "ENTAILMENT" and r["score"] >= e_thresh]
        if contradicting:
            statuses.append("CONTRADICTED")
        elif entailing:
            statuses.append("SUPPORTED")
        else:
            statuses.append("UNRESOLVED")
    if any(s == "CONTRADICTED" for s in statuses):
        return "CONTRADICTORY"
    if any(s == "UNRESOLVED" for s in statuses):
        return "INSUFFICIENT_EVIDENCE"
    return "CONSISTENT"


def metrics(pairs):
    """pairs: list of (gold, pred). Returns accuracy, macro_f1, per-class, confusion."""
    total = len(pairs)
    correct = sum(1 for g, p in pairs if g == p)
    accuracy = correct / total if total else 0.0
    per_class = {}
    f1s = []
    for v in VERDICTS:
        tp = sum(1 for g, p in pairs if g == v and p == v)
        fp = sum(1 for g, p in pairs if g != v and p == v)
        fn = sum(1 for g, p in pairs if g == v and p != v)
        support = sum(1 for g, _ in pairs if g == v)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class[v] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)
    macro_p = sum(per_class[v]["precision"] for v in VERDICTS) / len(VERDICTS)
    macro_r = sum(per_class[v]["recall"] for v in VERDICTS) / len(VERDICTS)
    return accuracy, macro_f1, macro_p, macro_r, per_class


def main() -> int:
    dump = json.loads(NLI_DUMP_PATH.read_text())
    labeled = [r for r in dump if r.get("gold_label") in GOLD]
    print(f"Rows with gold labels: {len(labeled)}")

    # --- raw NLI score distribution diagnostics ---
    contra_scores, entail_scores = [], []
    for r in dump:
        for cid, results in r["nli_by_claim"].items():
            for x in results:
                if x["label"] == "CONTRADICTION":
                    contra_scores.append(x["score"])
                elif x["label"] == "ENTAILMENT":
                    entail_scores.append(x["score"])

    def dist(scores, name):
        if not scores:
            print(f"  {name}: none")
            return
        scores = sorted(scores)
        n = len(scores)
        buckets = Counter()
        for s in scores:
            buckets[f"{int(s*10)/10:.1f}"] += 1
        print(f"  {name}: n={n} min={scores[0]:.3f} median={scores[n//2]:.3f} max={scores[-1]:.3f}")
        print(f"    by-0.1-bucket: {dict(sorted(buckets.items()))}")

    print("NLI score distribution (across ALL evidence pairs, with-graph train):")
    dist(contra_scores, "CONTRADICTION")
    dist(entail_scores, "ENTAILMENT")
    print()

    # --- baseline (current production thresholds) ---
    base_pairs = [(GOLD[r["gold_label"]], verdict_for_row(r["nli_by_claim"], r["claim_ids"], 0.5, 0.5)) for r in labeled]
    acc, mf1, mp, mr, pc = metrics(base_pairs)
    print(f"CURRENT thresholds (c=0.50, e=0.50): accuracy={acc:.4f} macro_f1={mf1:.4f} macro_p={mp:.4f} macro_r={mr:.4f}")
    print(f"  per-class: {json.dumps({v: {k: round(x,3) for k,x in pc[v].items()} for v in VERDICTS})}")
    print()

    # --- sweep ---
    c_grid = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]
    e_grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    results = []
    for c in c_grid:
        for e in e_grid:
            pairs = [(GOLD[r["gold_label"]], verdict_for_row(r["nli_by_claim"], r["claim_ids"], c, e)) for r in labeled]
            acc, mf1, mp, mr, _ = metrics(pairs)
            results.append((c, e, acc, mf1, mp, mr))

    print("=== SWEEP: top 15 by accuracy ===")
    for c, e, acc, mf1, mp, mr in sorted(results, key=lambda x: (-x[2], -x[3]))[:15]:
        print(f"  c={c:.2f} e={e:.2f}: accuracy={acc:.4f} macro_f1={mf1:.4f} macro_p={mp:.4f} macro_r={mr:.4f}")
    print("\n=== SWEEP: top 15 by macro_f1 ===")
    for c, e, acc, mf1, mp, mr in sorted(results, key=lambda x: (-x[3], -x[2]))[:15]:
        print(f"  c={c:.2f} e={e:.2f}: accuracy={acc:.4f} macro_f1={mf1:.4f} macro_p={mp:.4f} macro_r={mr:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
