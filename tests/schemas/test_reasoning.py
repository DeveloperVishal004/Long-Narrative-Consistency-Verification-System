"""ReasoningStep and LedgerEvent validation tests."""

from datetime import datetime, timezone

from lncvs.schemas import LedgerEvent, PipelineStage, ReasoningStep


def test_reasoning_step_valid_construction() -> None:
    step = ReasoningStep(
        stage=PipelineStage.RETRIEVAL,
        description="Retrieved 5 semantic and 5 lexical candidates.",
        timestamp=datetime.now(timezone.utc),
    )
    assert step.stage is PipelineStage.RETRIEVAL


def test_ledger_event_valid_construction() -> None:
    event = LedgerEvent(
        event_id="evt-1",
        stage=PipelineStage.RULE_ENGINE,
        message="Final verdict set: CONTRADICTORY",
        timestamp=datetime.now(timezone.utc),
    )
    assert event.message.startswith("Final verdict")
