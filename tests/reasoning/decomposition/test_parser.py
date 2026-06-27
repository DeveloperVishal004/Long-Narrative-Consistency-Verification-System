"""Decomposition parser tests — pure, no LLM involved."""

import pytest

from lncvs.reasoning.decomposition import parse_decomposition_response

PARENT_ID = "parent-1"


def test_valid_json_array_produces_correctly_indexed_atomic_claims() -> None:
    raw = '["John played piano", "John used both hands", "the event occurred in London"]'

    claims = parse_decomposition_response(raw, PARENT_ID, max_atomic_claims=10)

    assert [c.text for c in claims] == [
        "John played piano",
        "John used both hands",
        "the event occurred in London",
    ]
    assert [c.index for c in claims] == [0, 1, 2]
    assert all(c.parent_claim_id == PARENT_ID for c in claims)


def test_malformed_json_raises() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_decomposition_response("not json at all", PARENT_ID, max_atomic_claims=10)


def test_non_array_json_raises() -> None:
    with pytest.raises(ValueError, match="JSON array of strings"):
        parse_decomposition_response('{"not": "an array"}', PARENT_ID, max_atomic_claims=10)


def test_array_with_non_string_element_raises() -> None:
    with pytest.raises(ValueError, match="JSON array of strings"):
        parse_decomposition_response('["valid", 42]', PARENT_ID, max_atomic_claims=10)


def test_empty_array_raises() -> None:
    with pytest.raises(ValueError, match="zero atomic claims"):
        parse_decomposition_response("[]", PARENT_ID, max_atomic_claims=10)


def test_array_of_only_whitespace_raises() -> None:
    with pytest.raises(ValueError, match="zero atomic claims"):
        parse_decomposition_response('["   ", ""]', PARENT_ID, max_atomic_claims=10)


def test_duplicate_texts_are_deduplicated_case_insensitively() -> None:
    raw = '["John played piano", "john played piano", "John used both hands"]'

    claims = parse_decomposition_response(raw, PARENT_ID, max_atomic_claims=10)

    assert len(claims) == 2
    assert [c.text for c in claims] == ["John played piano", "John used both hands"]


def test_whitespace_around_claims_is_stripped() -> None:
    raw = '["  John played piano  ", "John used both hands"]'

    claims = parse_decomposition_response(raw, PARENT_ID, max_atomic_claims=10)

    assert claims[0].text == "John played piano"


def test_exceeding_max_atomic_claims_raises() -> None:
    raw = '["claim one", "claim two", "claim three"]'

    with pytest.raises(ValueError, match="max_atomic_claims"):
        parse_decomposition_response(raw, PARENT_ID, max_atomic_claims=2)


def test_claim_ids_are_deterministic_across_calls() -> None:
    raw = '["John played piano"]'

    first = parse_decomposition_response(raw, PARENT_ID, max_atomic_claims=10)
    second = parse_decomposition_response(raw, PARENT_ID, max_atomic_claims=10)

    assert first[0].claim_id == second[0].claim_id


def test_markdown_json_fence_is_stripped() -> None:
    """Real-execution finding (Phase H1): gemini-2.5-flash wraps the JSON
    array in a ```json ... ``` fence in ~29% of real decomposition calls
    despite the prompt's 'JSON array only' instruction. Must not raise."""
    raw = '```json\n["John played piano", "John used both hands"]\n```'

    claims = parse_decomposition_response(raw, PARENT_ID, max_atomic_claims=10)

    assert [c.text for c in claims] == ["John played piano", "John used both hands"]


def test_markdown_fence_without_json_language_tag_is_stripped() -> None:
    raw = '```\n["John played piano"]\n```'

    claims = parse_decomposition_response(raw, PARENT_ID, max_atomic_claims=10)

    assert claims[0].text == "John played piano"


def test_unwrapped_json_is_unaffected_by_fence_stripping() -> None:
    """The common, correctly-unwrapped case must remain a pure no-op."""
    raw = '["John played piano"]'

    claims = parse_decomposition_response(raw, PARENT_ID, max_atomic_claims=10)

    assert claims[0].text == "John played piano"
