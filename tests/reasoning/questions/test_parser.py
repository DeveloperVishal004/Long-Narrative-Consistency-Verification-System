"""Question generation parser tests — pure, no LLM involved."""

import pytest

from lncvs.reasoning.questions import parse_question_response

CLAIM_ID = "claim-1"


def test_valid_json_array_produces_correctly_indexed_questions() -> None:
    raw = '["Did John lose an arm?", "Did John suffer an injury?"]'

    questions = parse_question_response(raw, CLAIM_ID, max_questions_per_claim=10)

    assert [q.text for q in questions] == ["Did John lose an arm?", "Did John suffer an injury?"]
    assert [q.index for q in questions] == [0, 1]
    assert all(q.atomic_claim_id == CLAIM_ID for q in questions)


def test_empty_array_is_valid_and_returns_empty_list() -> None:
    questions = parse_question_response("[]", CLAIM_ID, max_questions_per_claim=10)
    assert questions == []


def test_malformed_json_raises() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_question_response("not json at all", CLAIM_ID, max_questions_per_claim=10)


def test_non_array_json_raises() -> None:
    with pytest.raises(ValueError, match="JSON array of strings"):
        parse_question_response('{"not": "an array"}', CLAIM_ID, max_questions_per_claim=10)


def test_array_with_non_string_element_raises() -> None:
    with pytest.raises(ValueError, match="JSON array of strings"):
        parse_question_response('["Did John lose an arm?", 42]', CLAIM_ID, max_questions_per_claim=10)


def test_exceeding_max_questions_per_claim_raises() -> None:
    raw = '["Question one?", "Question two?", "Question three?"]'
    with pytest.raises(ValueError, match="max_questions_per_claim"):
        parse_question_response(raw, CLAIM_ID, max_questions_per_claim=2)


def test_duplicate_questions_are_deduplicated_case_insensitively() -> None:
    raw = '["Did John lose an arm?", "did john lose an arm?", "Did John suffer an injury?"]'

    questions = parse_question_response(raw, CLAIM_ID, max_questions_per_claim=10)

    assert len(questions) == 2


def test_whitespace_around_questions_is_stripped() -> None:
    raw = '["  Did John lose an arm?  "]'

    questions = parse_question_response(raw, CLAIM_ID, max_questions_per_claim=10)

    assert questions[0].text == "Did John lose an arm?"


def test_question_ids_are_deterministic_across_calls() -> None:
    raw = '["Did John lose an arm?"]'

    first = parse_question_response(raw, CLAIM_ID, max_questions_per_claim=10)
    second = parse_question_response(raw, CLAIM_ID, max_questions_per_claim=10)

    assert first[0].question_id == second[0].question_id


# --- Faithfulness proxy: declarative statements (asserting new facts) are filtered ---


def test_declarative_statement_is_filtered_out() -> None:
    """A declarative statement asserts a new fact as already true and must be dropped,
    not returned as if it were a legitimate probe question."""
    raw = '["John lost his left arm in 2010."]'

    questions = parse_question_response(raw, CLAIM_ID, max_questions_per_claim=10)

    assert questions == []


def test_mixed_response_keeps_only_the_question_shaped_entries() -> None:
    raw = '["Did John lose an arm?", "John lost his left arm in 2010.", "Did John suffer an injury?"]'

    questions = parse_question_response(raw, CLAIM_ID, max_questions_per_claim=10)

    assert [q.text for q in questions] == ["Did John lose an arm?", "Did John suffer an injury?"]


def test_all_declarative_response_returns_empty_list_not_an_error() -> None:
    """If every candidate fails the question-shape filter, the legitimate outcome is
    an empty result (per the empty-results-are-valid contract), not a raised error."""
    raw = '["John lost his left arm in 2010.", "John moved to London in 2012."]'

    questions = parse_question_response(raw, CLAIM_ID, max_questions_per_claim=10)

    assert questions == []
