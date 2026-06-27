"""Chapter segmentation: detection, fallback windowing, long-chapter
splitting, paragraph snapping, full-coverage invariant, determinism, and a
real-data sanity check against both project novels."""

from pathlib import Path

import pytest

from lncvs.graph.segmentation import (
    MAX_EXTRACTION_TOKENS,
    MIN_CHAPTERS,
    count_tokens,
    segment_into_chapters,
    segment_into_extraction_windows,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _assert_full_coverage_no_gaps(windows: list, text_length: int) -> None:
    ordered = sorted(windows, key=lambda w: w.char_start)
    assert ordered[0].char_start == 0
    assert ordered[-1].char_end == text_length
    for previous, current in zip(ordered, ordered[1:]):
        assert current.char_start <= previous.char_end, "gap between consecutive windows"


def test_count_tokens_is_deterministic() -> None:
    text = "John lost his left arm in an accident in 2010."
    assert count_tokens(text) == count_tokens(text)
    assert count_tokens(text) > 0


def test_detects_roman_numeral_chapter_headings() -> None:
    text = "\n\n".join(f"Chapter {roman}.\n\nSome chapter text here, paragraph one." for roman in ["I", "II", "III", "IV"])
    chapters = segment_into_chapters(text)
    assert len(chapters) == 4
    assert [c.chapter_index for c in chapters] == [0, 1, 2, 3]


def test_detects_arabic_chapter_headings_with_trailing_title() -> None:
    text = "\n\n".join(f"Chapter {n}. Some Title Here\n\nBody text for this chapter." for n in range(1, 5))
    chapters = segment_into_chapters(text)
    assert len(chapters) == 4


def test_preamble_before_first_heading_is_kept_as_chapter_zero() -> None:
    text = "PREFACE\n\nIntroductory remarks.\n\n" + "\n\n".join(
        f"Chapter {n}.\n\nBody text." for n in range(1, 4)
    )
    chapters = segment_into_chapters(text)
    assert chapters[0].chapter_index == 0
    assert chapters[0].char_start == 0
    assert text[chapters[0].char_start : chapters[0].char_end].startswith("PREFACE")


def test_no_preamble_emitted_when_first_heading_is_at_offset_zero() -> None:
    text = "\n\n".join(f"Chapter {n}.\n\nBody text." for n in range(1, 4))
    chapters = segment_into_chapters(text)
    assert chapters[0].char_start == 0
    assert text[chapters[0].char_start : chapters[0].char_end].startswith("Chapter 1")


def test_falls_back_to_fixed_windows_when_no_chapter_headings_exist() -> None:
    text = "This is plain prose with no chapter markers at all. " * 50
    windows = segment_into_extraction_windows(text)
    assert len(windows) >= 1
    _assert_full_coverage_no_gaps(windows, len(text))


def test_falls_back_when_fewer_than_min_chapters_detected() -> None:
    assert MIN_CHAPTERS == 3
    text = "Chapter 1.\n\nBody one. " + "Chapter 2.\n\nBody two. " * 1
    chapters = segment_into_chapters(text)
    # Only 2 headings present -- below MIN_CHAPTERS -- so this must NOT be
    # treated as real chapter structure; it falls back to one fixed window
    # (the text is short enough to fit in a single fallback window).
    assert len(chapters) == 1


def test_long_chapter_is_split_into_overlapping_sub_windows() -> None:
    long_chapter_body = ("Paragraph of narrative text padding out the chapter length. " * 400) + "\n\n"
    text = "Chapter 1.\n\n" + long_chapter_body * 3 + "Chapter 2.\n\nShort body.\n\nChapter 3.\n\nShort body."
    windows = segment_into_extraction_windows(text)

    chapter_zero_windows = [w for w in windows if w.chapter_index == 0]
    assert len(chapter_zero_windows) > 1
    assert all(w.window_index is not None for w in chapter_zero_windows)
    assert [w.window_index for w in chapter_zero_windows] == sorted(w.window_index for w in chapter_zero_windows)


def test_extraction_windows_cover_entire_text_with_no_gaps() -> None:
    long_chapter_body = ("Paragraph of narrative text padding out the chapter length. " * 400) + "\n\n"
    text = "Chapter 1.\n\n" + long_chapter_body * 3 + "Chapter 2.\n\nShort body.\n\nChapter 3.\n\nShort body."
    windows = segment_into_extraction_windows(text)
    _assert_full_coverage_no_gaps(windows, len(text))


def test_compute_window_boundaries_snaps_when_a_break_is_in_range_and_within_budget() -> None:
    from lncvs.graph.segmentation import _compute_window_boundaries

    # Short paragraphs (~12 tokens each) every ~50 characters -- a paragraph
    # break is always well within the 200-token snap radius of any
    # token-count boundary, and snapping to the nearest one costs only a
    # few tokens, comfortably within budget.
    span_text = ("Paragraph text padding the chapter body out. \n\n" * 400).rstrip()

    boundaries = _compute_window_boundaries(
        span_text, max_tokens=400, overlap_tokens=60, snap_tolerance_tokens=200
    )
    assert len(boundaries) > 1
    # Best-effort, not guaranteed for every boundary: the nearest paragraph
    # break sometimes overshoots the token budget by a few tokens (see the
    # next test), in which case the hard budget wins and that one boundary
    # is left unsnapped. At least one boundary in a densely-paragraphed
    # text like this one should still land cleanly on a break.
    assert any(span_text[end - 2 : end] == "\n\n" for _, end in boundaries[:-1])


def test_compute_window_boundaries_never_exceeds_max_tokens_even_when_no_break_is_nearby() -> None:
    from lncvs.graph.segmentation import _compute_window_boundaries, count_tokens

    # No paragraph breaks at all near any boundary -- snapping must yield
    # to the hard token budget rather than overshoot it.
    span_text = "Sentence padding the chapter with no paragraph breaks whatsoever. " * 400

    boundaries = _compute_window_boundaries(
        span_text, max_tokens=400, overlap_tokens=60, snap_tolerance_tokens=200
    )
    assert len(boundaries) > 1
    for start, end in boundaries:
        assert count_tokens(span_text[start:end]) <= 400


def test_segmentation_is_deterministic_across_calls() -> None:
    text = "\n\n".join(f"Chapter {n}.\n\nBody text padding {n}." for n in range(1, 6))
    first = segment_into_extraction_windows(text)
    second = segment_into_extraction_windows(text)
    assert [(w.chapter_index, w.window_index, w.char_start, w.char_end) for w in first] == [
        (w.chapter_index, w.window_index, w.char_start, w.char_end) for w in second
    ]


def test_segment_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        segment_into_chapters("")
    with pytest.raises(ValueError):
        segment_into_extraction_windows("")


# --- Real-data sanity check (fast: pure regex + tiktoken, no model loading) ---


@pytest.mark.parametrize(
    "filename,min_expected_chapters",
    [
        ("In search of the castaways.txt", 50),
        ("The Count of Monte Cristo.txt", 100),
    ],
)
def test_real_novel_chapter_detection_and_full_coverage(filename: str, min_expected_chapters: int) -> None:
    from lncvs.ingestion import load_and_clean_narrative

    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not present in data/")

    document = load_and_clean_narrative(path, source_id=filename)
    chapters = segment_into_chapters(document.cleaned_text)
    assert len(chapters) >= min_expected_chapters

    windows = segment_into_extraction_windows(document.cleaned_text)
    _assert_full_coverage_no_gaps(windows, len(document.cleaned_text))
    assert all(
        count_tokens(document.cleaned_text[w.char_start : w.char_end]) <= MAX_EXTRACTION_TOKENS for w in windows
    )
