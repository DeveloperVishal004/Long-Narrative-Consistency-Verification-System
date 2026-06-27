"""EvidenceLedger validation tests.

These tests assert the ledger's defining property: every field is a typed
model or a typed list of models, never a raw dict or List[Any]. Pydantic
will still coerce a well-shaped dict into the correct nested model (that's
normal validation, not a violation) — what must fail is a malformed or
wrong-shaped dict.
"""

import pytest
from pydantic import ValidationError

from lncvs.schemas import AtomicClaim, EvidenceLedger


def test_empty_ledger_constructs_with_defaults() -> None:
    ledger = EvidenceLedger(original_claim="John played a two-handed piano piece in London.")
    assert ledger.atomic_claims == []
    assert ledger.nli_results == []
    assert ledger.final_verdict is None
    assert ledger.original_claim_id is None
    assert ledger.retrieval_queries == []


def test_ledger_accepts_explicit_original_claim_id() -> None:
    ledger = EvidenceLedger(
        original_claim="John played a two-handed piano piece in London.",
        original_claim_id="abc123",
    )
    assert ledger.original_claim_id == "abc123"


def test_ledger_rejects_empty_original_claim() -> None:
    with pytest.raises(ValidationError):
        EvidenceLedger(original_claim="")


def test_ledger_accepts_well_shaped_dict_for_nested_model() -> None:
    """A dict matching AtomicClaim's shape is validated into a real AtomicClaim, not stored as a dict."""
    ledger = EvidenceLedger(
        original_claim="John played a two-handed piano piece in London.",
        atomic_claims=[{"claim_id": "claim-1", "text": "John used both hands"}],
    )
    assert isinstance(ledger.atomic_claims[0], AtomicClaim)


def test_ledger_rejects_malformed_nested_dict() -> None:
    """A dict missing AtomicClaim's required fields must fail validation, not pass through silently."""
    with pytest.raises(ValidationError):
        EvidenceLedger(
            original_claim="John played a two-handed piano piece in London.",
            atomic_claims=[{"unexpected_field": "no claim_id or text here"}],
        )


def test_ledger_final_verdict_defaults_to_none_until_rule_engine_sets_it() -> None:
    ledger = EvidenceLedger(original_claim="John played a two-handed piano piece in London.")
    assert ledger.final_verdict is None
