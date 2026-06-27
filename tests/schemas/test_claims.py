"""AtomicClaim and ProbeQuestion validation tests."""

import pytest
from pydantic import ValidationError

from lncvs.schemas import AtomicClaim, ProbeQuestion


def test_atomic_claim_valid_construction() -> None:
    claim = AtomicClaim(claim_id="claim-1", text="John used both hands", parent_claim_id="claim-0")
    assert claim.parent_claim_id == "claim-0"


def test_atomic_claim_parent_is_optional() -> None:
    claim = AtomicClaim(claim_id="claim-1", text="John used both hands")
    assert claim.parent_claim_id is None


def test_atomic_claim_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        AtomicClaim(claim_id="claim-1", text="")


def test_atomic_claim_index_defaults_to_zero() -> None:
    claim = AtomicClaim(claim_id="claim-1", text="John used both hands")
    assert claim.index == 0


def test_atomic_claim_accepts_explicit_index() -> None:
    claim = AtomicClaim(claim_id="claim-1", text="John used both hands", index=2)
    assert claim.index == 2


def test_atomic_claim_rejects_negative_index() -> None:
    with pytest.raises(ValidationError):
        AtomicClaim(claim_id="claim-1", text="John used both hands", index=-1)


def test_probe_question_valid_construction() -> None:
    question = ProbeQuestion(
        question_id="q-1", atomic_claim_id="claim-1", text="Did John lose an arm?"
    )
    assert question.atomic_claim_id == "claim-1"


def test_probe_question_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        ProbeQuestion(question_id="q-1", atomic_claim_id="claim-1", text="")


def test_probe_question_index_defaults_to_zero() -> None:
    question = ProbeQuestion(question_id="q-1", atomic_claim_id="claim-1", text="Did John lose an arm?")
    assert question.index == 0


def test_probe_question_accepts_explicit_index() -> None:
    question = ProbeQuestion(
        question_id="q-1", atomic_claim_id="claim-1", text="Did John lose an arm?", index=1
    )
    assert question.index == 1


def test_probe_question_rejects_negative_index() -> None:
    with pytest.raises(ValidationError):
        ProbeQuestion(question_id="q-1", atomic_claim_id="claim-1", text="Did John lose an arm?", index=-1)
