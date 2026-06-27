"""compute_latency_metrics() tests."""

from datetime import datetime, timedelta, timezone

from lncvs.evaluation.metrics.latency import compute_latency_metrics
from lncvs.schemas import EvidenceLedger, LedgerEvent, PipelineStage


def _event(stage: PipelineStage, message: str, timestamp: datetime) -> LedgerEvent:
    return LedgerEvent(event_id=f"event-{timestamp.isoformat()}", stage=stage, message=message, timestamp=timestamp)


def test_returns_zero_latency_for_empty_log() -> None:
    ledger = EvidenceLedger(original_claim="claim")
    metrics = compute_latency_metrics(ledger)

    assert metrics.stages == []
    assert metrics.end_to_end_ms == 0.0


def test_end_to_end_spans_earliest_to_latest_event() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ledger = EvidenceLedger(original_claim="claim")
    ledger.ledger_log.extend(
        [
            _event(PipelineStage.CLAIM_DECOMPOSITION, "step 1", base),
            _event(PipelineStage.RETRIEVAL, "step 2", base + timedelta(milliseconds=250)),
        ]
    )

    metrics = compute_latency_metrics(ledger)

    assert metrics.end_to_end_ms == 250.0


def test_per_stage_duration_spans_that_stage_own_events() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ledger = EvidenceLedger(original_claim="claim")
    ledger.ledger_log.extend(
        [
            _event(PipelineStage.RETRIEVAL, "query 1", base),
            _event(PipelineStage.RETRIEVAL, "query 2", base + timedelta(milliseconds=100)),
            _event(PipelineStage.FUSION, "fuse", base + timedelta(milliseconds=150)),
        ]
    )

    metrics = compute_latency_metrics(ledger)

    by_stage = {stage.stage: stage.duration_ms for stage in metrics.stages}
    assert by_stage[PipelineStage.RETRIEVAL] == 100.0
    assert by_stage[PipelineStage.FUSION] == 0.0
