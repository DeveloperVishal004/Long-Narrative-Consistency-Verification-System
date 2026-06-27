"""norm_name: the sole basis for merge decisions, plus is_generic_referent:
the guard that keeps non-identifying generic referents from bridging."""

import pytest

from lncvs.graph.entity_resolution.normalization import is_generic_referent, norm_name


def test_lowercases() -> None:
    assert norm_name("John") == norm_name("JOHN") == norm_name("john")


def test_strips_single_leading_title() -> None:
    assert norm_name("Lord Glenarvan") == norm_name("Glenarvan")


def test_strips_multiple_leading_titles() -> None:
    assert norm_name("The Captain John") == norm_name("John")


def test_strips_punctuation_and_leading_title_abbreviation() -> None:
    assert norm_name("M. Dantès") == norm_name("Dantès") == "dantès"


def test_strips_period_after_abbreviated_title() -> None:
    assert norm_name("Dr. Watson") == norm_name("Watson")


def test_collapses_whitespace() -> None:
    assert norm_name("John   Smith") == norm_name("John Smith")


def test_distinct_names_remain_distinct() -> None:
    assert norm_name("John") != norm_name("Mary")


def test_empty_or_title_only_name_normalizes_to_empty() -> None:
    assert norm_name("The") == ""
    assert norm_name("") == ""


def test_is_deterministic() -> None:
    assert norm_name("Lord Glenarvan") == norm_name("Lord Glenarvan")


def test_expanded_titles_strip_rank_and_nobility() -> None:
    assert norm_name("Major MacNabb") == norm_name("MacNabb") == "macnabb"
    assert norm_name("King Louis XVI") == norm_name("Louis XVI") == "louis xvi"
    assert norm_name("General d'Épinay") == norm_name("d'Épinay")


@pytest.mark.parametrize(
    "name",
    [
        "he", "him", "his", "she", "her", "they", "this man", "the man",
        "the major", "chief", "my father", "his son", "the geographer",
        "young man", "old man", "your excellency", "the traveller", "the assassin",
        "the yacht", "",
    ],
)
def test_generic_referents_are_flagged(name: str) -> None:
    assert is_generic_referent(norm_name(name)) is True


@pytest.mark.parametrize(
    "name",
    [
        "Paganel", "Glenarvan", "Ayrton", "Ben Joyce", "Major MacNabb",
        "Louis XVI", "Edmond Dantès", "young Dantès", "little Edward",
        "Captain Grant",
    ],
)
def test_specific_names_are_not_flagged(name: str) -> None:
    assert is_generic_referent(norm_name(name)) is False
