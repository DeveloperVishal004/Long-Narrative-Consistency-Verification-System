"""Latency metrics derived from the ledger's append-only audit log.

Latency is the one metric NOT subject to the determinism guarantee -- per
CLAUDE.md, wall-clock fields (ledger_log timestamps) are an accepted
exception to full ledger reproducibility.
"""

from lncvs.schemas import EvidenceLedger
from lncvs.schemas.evaluation import LatencyMetrics, StageLatency


def compute_latency_metrics(ledger: EvidenceLedger) -> LatencyMetrics:
    """Per-stage duration spans the earliest-to-latest timestamp logged for
    that stage; this is a coarse approximation when a stage logs only one
    event (its span collapses to 0ms), which is the common case today."""
    if not ledger.ledger_log:
        return LatencyMetrics(stages=[], end_to_end_ms=0.0)

    timestamps_by_stage: dict = {}
    for event in ledger.ledger_log:
        timestamps_by_stage.setdefault(event.stage, []).append(event.timestamp)

    stages: list[StageLatency] = []
    for stage, timestamps in timestamps_by_stage.items():
        duration_ms = (max(timestamps) - min(timestamps)).total_seconds() * 1000.0
        stages.append(StageLatency(stage=stage, duration_ms=duration_ms))

    all_timestamps = [event.timestamp for event in ledger.ledger_log]
    end_to_end_ms = (max(all_timestamps) - min(all_timestamps)).total_seconds() * 1000.0

    return LatencyMetrics(stages=stages, end_to_end_ms=end_to_end_ms)
