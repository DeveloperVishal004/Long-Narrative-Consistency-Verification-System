"""ControlState and GraphState validation tests, including the ledger/control separation."""

from lncvs.schemas import ControlState, EvidenceLedger, GraphState, PipelineStage


def test_control_state_valid_construction() -> None:
    control = ControlState(current_stage=PipelineStage.RETRIEVAL, config_fingerprint="abc123")
    assert control.retry_count == 0
    assert control.errors == []
    assert control.degraded_sources == []


def test_graph_state_keeps_ledger_and_control_separate() -> None:
    ledger = EvidenceLedger(original_claim="John played a two-handed piano piece in London.")
    control = ControlState(current_stage=PipelineStage.INGESTION, config_fingerprint="abc123")
    state = GraphState(ledger=ledger, control=control)

    assert state.ledger is ledger
    assert state.control is control
    # The ledger must carry no orchestration-only fields, and control must carry no domain fields.
    assert not hasattr(state.ledger, "current_stage")
    assert not hasattr(state.control, "final_verdict")
