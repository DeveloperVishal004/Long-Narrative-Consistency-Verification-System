"""Deterministic mention extraction: stopword filtering, multi-word merging,
order/dedup, min-length filtering."""

from lncvs.graph.extraction import extract_mentions


def test_extracts_capitalized_single_word_mentions() -> None:
    mentions = extract_mentions("John lost his left arm in an accident in 2010.", min_token_length=2)
    assert mentions == ["John"]


def test_extracts_multiple_distinct_mentions_in_order_of_first_appearance() -> None:
    mentions = extract_mentions("John moved to London in 2012.", min_token_length=2)
    assert mentions == ["John", "London"]


def test_merges_consecutive_capitalized_tokens_into_one_mention() -> None:
    mentions = extract_mentions("She visited New York last year.", min_token_length=2)
    assert mentions == ["New York"]


def test_filters_stopwords() -> None:
    mentions = extract_mentions("The dog ran. It was fast.", min_token_length=2)
    assert mentions == []


def test_filters_below_min_token_length() -> None:
    mentions = extract_mentions("Ed went home.", min_token_length=3)
    assert mentions == []


def test_deduplicates_repeated_mentions() -> None:
    mentions = extract_mentions("John ran. Then John walked.", min_token_length=2)
    assert mentions == ["John"]


def test_is_deterministic_across_calls() -> None:
    text = "Paganel and Glenarvan sailed the Duncan to London."
    assert extract_mentions(text, min_token_length=2) == extract_mentions(text, min_token_length=2)
