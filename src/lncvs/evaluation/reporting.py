"""JSON persistence and matplotlib visualization for evaluation reports.

No external experiment tracker (MLflow, W&B, etc.) is used -- reports are
plain JSON files on the local filesystem, keyed by the deterministic run_id
already embedded in each EvaluationReport.
"""

import logging
from pathlib import Path

from lncvs.schemas import EvidenceLedger, VerdictEnum
from lncvs.schemas.evaluation import AblationReport, EvaluationReport

logger = logging.getLogger(__name__)


def save_report(report: EvaluationReport, output_dir: Path) -> Path:
    """Persist report as JSON, keyed by its deterministic run_id."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{report.run_id}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved evaluation report %s to %s", report.run_id, path)
    return path


def save_ledger(ledger: EvidenceLedger, output_dir: Path, filename: str) -> Path:
    """Persist a full EvidenceLedger as JSON, for on-demand debugging.

    EvaluationReport itself never embeds the ledger body (see
    schemas/evaluation.py's module docstring) -- this is the opt-in escape
    hatch, called only when EvaluationConfig.persist_ledgers is True, with
    ExampleResult.ledger_path pointing back to the file this writes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename}.json"
    path.write_text(ledger.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved ledger to %s", path)
    return path


def plot_ablation_deltas(report: AblationReport, output_path: Path) -> Path:
    """Bar chart of each ablated component's contribution delta on verdict accuracy."""
    if not report.deltas:
        raise ValueError("Cannot plot an AblationReport with no contribution deltas")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [delta.component.value for delta in report.deltas]
    values = [delta.delta for delta in report.deltas]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Accuracy delta (full - ablated)")
    ax.set_title("Component contribution to verdict accuracy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(report: EvaluationReport, output_path: Path) -> Path:
    """Heatmap of the report's verdict confusion matrix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verdicts = list(VerdictEnum)
    index = {verdict: i for i, verdict in enumerate(verdicts)}
    matrix = [[0 for _ in verdicts] for _ in verdicts]
    for cell in report.verdict.confusion.cells:
        matrix[index[cell.gold]][index[cell.predicted]] = cell.count

    fig, ax = plt.subplots()
    ax.imshow(matrix)
    ax.set_xticks(range(len(verdicts)), [verdict.value for verdict in verdicts], rotation=45, ha="right")
    ax.set_yticks(range(len(verdicts)), [verdict.value for verdict in verdicts])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    for row in range(len(verdicts)):
        for col in range(len(verdicts)):
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
