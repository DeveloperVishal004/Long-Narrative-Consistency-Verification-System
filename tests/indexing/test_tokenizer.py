"""Shared tokenizer tests."""

from lncvs.indexing import tokenize


def test_tokenize_lowercases() -> None:
    assert tokenize("John Lost His Arm") == ["john", "lost", "his", "arm"]


def test_tokenize_strips_punctuation() -> None:
    assert tokenize("Did John lose an arm?") == ["did", "john", "lose", "an", "arm"]


def test_tokenize_is_deterministic() -> None:
    text = "John lost his left arm in an accident in 2010."
    assert tokenize(text) == tokenize(text)


def test_tokenize_empty_string_returns_empty_list() -> None:
    assert tokenize("") == []


def test_tokenize_punctuation_only_returns_empty_list() -> None:
    assert tokenize("???!!!") == []


def test_tokenize_preserves_numbers() -> None:
    assert tokenize("John moved to London in 2012.") == ["john", "moved", "to", "london", "in", "2012"]
