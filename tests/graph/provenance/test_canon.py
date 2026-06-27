"""canonicalize_with_offsets: the shared transform + offset-recovery
mechanism both sides of every quote match go through."""

from lncvs.graph.provenance.canon import canonicalize_with_offsets


def _recover(text: str, canon_start: int, canon_end: int) -> str:
    canon_text, offsets = canonicalize_with_offsets(text)
    raw_start = offsets[canon_start]
    raw_end = offsets[canon_end - 1] + 1
    return text[raw_start:raw_end]


def test_identity_for_plain_ascii_text() -> None:
    canon_text, offsets = canonicalize_with_offsets("John lost his arm.")
    assert canon_text == "John lost his arm."
    assert offsets == list(range(len(canon_text)))


def test_smart_quotes_and_dashes_normalized() -> None:
    canon_text, _ = canonicalize_with_offsets("He said “hello—world”.")
    assert canon_text == 'He said "hello-world".'


def test_ellipsis_normalized() -> None:
    canon_text, _ = canonicalize_with_offsets("Wait… what?")
    assert canon_text == "Wait... what?"


def test_whitespace_runs_collapsed_to_single_space() -> None:
    canon_text, _ = canonicalize_with_offsets("John   lost\n\nhis\targm.")
    assert canon_text == "John lost his argm."


def test_leading_and_trailing_whitespace_stripped() -> None:
    canon_text, offsets = canonicalize_with_offsets("   John ran.   ")
    assert canon_text == "John ran."
    assert len(offsets) == len(canon_text)


def test_offsets_recover_exact_original_substring_through_normalization() -> None:
    original = "He said “I lost   my arm” in 2010."
    canon_text, offsets = canonicalize_with_offsets(original)
    idx = canon_text.find("lost my arm")
    recovered = _recover(original, idx, idx + len("lost my arm"))
    assert recovered == "lost   my arm"  # original whitespace preserved exactly


def test_empty_string_canonicalizes_to_empty() -> None:
    canon_text, offsets = canonicalize_with_offsets("")
    assert canon_text == ""
    assert offsets == []


def test_whitespace_only_string_canonicalizes_to_empty() -> None:
    canon_text, offsets = canonicalize_with_offsets("   \n\t  ")
    assert canon_text == ""
    assert offsets == []


def test_canonicalization_is_deterministic() -> None:
    text = "John said ‘hello’ to Mary—twice."
    assert canonicalize_with_offsets(text) == canonicalize_with_offsets(text)
