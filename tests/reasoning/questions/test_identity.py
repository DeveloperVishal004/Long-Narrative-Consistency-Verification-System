"""Identity (content-hash ID derivation) tests for question generation."""

from lncvs.reasoning.questions import make_question_id


def test_make_question_id_is_deterministic() -> None:
    assert make_question_id("claim-1", 0, "Did John lose an arm?") == make_question_id(
        "claim-1", 0, "Did John lose an arm?"
    )


def test_make_question_id_differs_by_index() -> None:
    id_a = make_question_id("claim-1", 0, "same text")
    id_b = make_question_id("claim-1", 1, "same text")
    assert id_a != id_b


def test_make_question_id_differs_by_text() -> None:
    id_a = make_question_id("claim-1", 0, "text A?")
    id_b = make_question_id("claim-1", 0, "text B?")
    assert id_a != id_b


def test_make_question_id_differs_by_atomic_claim_id() -> None:
    id_a = make_question_id("claim-1", 0, "same text")
    id_b = make_question_id("claim-2", 0, "same text")
    assert id_a != id_b


def test_no_collisions_within_a_set_of_questions_for_one_claim() -> None:
    texts = ["Did John lose an arm?", "Did John suffer an injury?", "What physical condition did John have?"]
    ids = [make_question_id("claim-1", index, text) for index, text in enumerate(texts)]
    assert len(set(ids)) == 3
