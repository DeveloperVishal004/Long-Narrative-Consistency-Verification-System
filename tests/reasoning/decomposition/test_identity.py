"""Identity (content-hash ID derivation) tests."""

from lncvs.reasoning.decomposition import make_atomic_claim_id, make_source_claim_id


def test_make_source_claim_id_is_deterministic() -> None:
    claim = "John played a two-handed piano piece in London."
    assert make_source_claim_id(claim) == make_source_claim_id(claim)


def test_make_source_claim_id_differs_for_different_text() -> None:
    assert make_source_claim_id("claim A") != make_source_claim_id("claim B")


def test_make_atomic_claim_id_is_deterministic() -> None:
    assert make_atomic_claim_id("parent-1", 0, "John played piano") == make_atomic_claim_id(
        "parent-1", 0, "John played piano"
    )


def test_make_atomic_claim_id_differs_by_index() -> None:
    id_a = make_atomic_claim_id("parent-1", 0, "same text")
    id_b = make_atomic_claim_id("parent-1", 1, "same text")
    assert id_a != id_b


def test_make_atomic_claim_id_differs_by_text() -> None:
    id_a = make_atomic_claim_id("parent-1", 0, "text A")
    id_b = make_atomic_claim_id("parent-1", 0, "text B")
    assert id_a != id_b


def test_make_atomic_claim_id_differs_by_parent() -> None:
    id_a = make_atomic_claim_id("parent-1", 0, "same text")
    id_b = make_atomic_claim_id("parent-2", 0, "same text")
    assert id_a != id_b


def test_dummy_case_trio_has_no_id_collisions() -> None:
    parent_id = make_source_claim_id("John played a two-handed piano piece in London.")
    texts = ["John played piano", "John used both hands", "the event occurred in London"]
    ids = [make_atomic_claim_id(parent_id, index, text) for index, text in enumerate(texts)]
    assert len(set(ids)) == 3
