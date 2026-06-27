"""resolve_quote: tiered exact / bounded-fuzzy / failed resolution."""

from lncvs.graph.provenance.config import ProvenanceConfig
from lncvs.graph.provenance.matching import MatchTier, resolve_quote

WINDOW = (
    "John lost his left arm in an accident in 2010. "
    "John moved to London in 2012. "
    "John lost his left arm in a terrible riding accident near the old stone bridge "
    "in the autumn of 2010 again, in a later retelling."
)


def test_exact_match_resolves_to_correct_span() -> None:
    match = resolve_quote("John moved to London in 2012.", WINDOW)
    assert match.tier is MatchTier.EXACT
    assert WINDOW[match.char_start : match.char_end] == "John moved to London in 2012."
    assert match.ambiguous is False


def test_repeated_quote_takes_first_occurrence_and_flags_ambiguous() -> None:
    """"John lost his left arm" appears twice in WINDOW (once in the short
    opening sentence, once inside the longer closing sentence)."""
    match = resolve_quote("John lost his left arm", WINDOW)
    assert match.tier is MatchTier.EXACT
    assert match.ambiguous is True
    first_occurrence = WINDOW.find("John lost his left arm")
    assert match.char_start == first_occurrence


def test_quote_not_found_returns_failed() -> None:
    match = resolve_quote("John traveled to Mars in a rocket.", WINDOW)
    assert match.tier is MatchTier.FAILED
    assert match.char_start is None
    assert match.char_end is None


def test_empty_or_whitespace_only_quote_fails() -> None:
    assert resolve_quote("   ", WINDOW).tier is MatchTier.FAILED


def test_fuzzy_fallback_succeeds_on_near_exact_quote_with_dropped_comma() -> None:
    # The real sentence (25 tokens) has "2010 again," with a comma; the
    # "quote" drops it -- a single token differs out of 25 (96% overlap),
    # comfortably clearing the 0.95 threshold, and the sentence is long
    # and specific enough to be the unique best candidate in the window.
    near_quote = (
        "John lost his left arm in a terrible riding accident near the old stone bridge "
        "in the autumn of 2010 again in a later retelling."
    )
    match = resolve_quote(near_quote, WINDOW)
    assert match.tier is MatchTier.FUZZY
    recovered = WINDOW[match.char_start : match.char_end]
    assert recovered.startswith("John lost his left arm in a terrible riding accident")


def test_fuzzy_fallback_rejects_when_overlap_too_low() -> None:
    unrelated = (
        "Mary sailed across the cold grey ocean for many long weeks to find her brother "
        "somewhere in the vast unfamiliar continent of Australia eventually."
    )
    match = resolve_quote(unrelated, WINDOW)
    assert match.tier is MatchTier.FAILED


def test_fuzzy_fallback_rejects_ambiguous_candidate() -> None:
    """Two near-identical long sentences in the window make any paraphrase
    of one an almost-equally-good fuzzy match for the other -- the
    uniqueness margin must reject this rather than guess."""
    window_with_two_near_duplicate_sentences = (
        "Long ago in a distant valley John lost his left arm in a terrible riding accident yesterday near the bridge. "
        "Long ago in a distant valley John lost his left arm in a terrible riding accident yesterday near the river."
    )
    near_quote = (
        "Long ago in a distant valley John lost his left arm in a terrible riding accident yesterday near the road."
    )
    config = ProvenanceConfig(fuzzy_overlap_threshold=0.80, fuzzy_uniqueness_margin=0.05)
    match = resolve_quote(near_quote, window_with_two_near_duplicate_sentences, config)
    assert match.tier is MatchTier.FAILED


def test_resolution_is_deterministic_across_calls() -> None:
    quote = "John moved to London in 2012."
    assert resolve_quote(quote, WINDOW) == resolve_quote(quote, WINDOW)
