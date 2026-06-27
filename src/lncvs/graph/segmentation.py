"""Chapter segmentation (Phase 8 / G2, frozen spec §1).

Deterministic, no LLM. Operates on the identical cleaned, offset-preserving
text RawDocument.cleaned_text already provides -- the same coordinate
space lncvs.chunking already chunks over -- so every offset this module
produces is directly comparable to a DocumentChunk's char_start/char_end
with no re-cleaning or re-encoding step in between (the frozen spec's
single cross-cutting correctness rule).

Two-level output:
  - ChapterSpan: one per detected chapter (or, in fallback mode, one per
    fixed-size segment) -- the natural narrative unit.
  - ExtractionWindow: the actual unit fed to extraction (Slice 3+). Equal
    to its ChapterSpan unless the chapter exceeds MAX_EXTRACTION_TOKENS,
    in which case it is split into overlapping sub-windows that share the
    same chapter_index and carry a window_index.

tiktoken is the only token-counting mechanism in this module, isolated the
same way chromadb/rank_bm25/openai are confined to one file each. Token
counts use the o200k_base encoding (the GPT-4o family's encoding), since
G2 extraction targets that model family and the budgets in this module
exist to keep extraction calls within that model's effective context.
"""

import re

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, model_validator

MIN_CHAPTERS = 3
MAX_EXTRACTION_TOKENS = 6000
FALLBACK_WINDOW_TOKENS = 4000
WINDOW_OVERLAP_TOKENS = 600
PARAGRAPH_SNAP_TOLERANCE_TOKENS = 200
_CHARS_PER_TOKEN_ESTIMATE = 4  # search-radius heuristic only; token counts themselves are always exact

_ENCODING = tiktoken.get_encoding("o200k_base")

_CHAPTER_HEADING_PATTERNS = [
    re.compile(r"^[ \t]*Chapter\s+[IVXLCDM]+\.?(?=\s|$)", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^[ \t]*Chapter\s+\d+\.?(?=\s|$)", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^[ \t]*Chapter\s+the\s+[A-Za-z]+\b", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^[ \t]*(?:[IVXLCDM]+|\d+)\.\s+[A-Z]", re.MULTILINE),
]


class ChapterSpan(BaseModel):
    """One detected chapter, or one fixed-size fallback segment when no
    chapter structure is detected. chapter_index is 0 for front-matter
    preamble (when chapters are detected) or runs sequentially across
    fallback segments."""

    model_config = ConfigDict(frozen=True)

    chapter_index: int = Field(..., ge=0)
    char_start: int = Field(..., ge=0)
    char_end: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _validate_span(self) -> "ChapterSpan":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be strictly greater than char_start")
        return self


class ExtractionWindow(BaseModel):
    """The unit actually fed to extraction (Slice 3+). window_index is
    None when this window is an entire chapter/fallback-segment
    unmodified; otherwise it is the 0-based sub-window index within that
    chapter, sharing chapter_index with its siblings."""

    model_config = ConfigDict(frozen=True)

    chapter_index: int = Field(..., ge=0)
    window_index: int | None = Field(default=None, ge=0)
    char_start: int = Field(..., ge=0)
    char_end: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _validate_span(self) -> "ExtractionWindow":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be strictly greater than char_start")
        return self


def count_tokens(text: str) -> int:
    """Exact token count under the o200k_base encoding. Deterministic."""
    return len(_ENCODING.encode(text))


def _char_offset_for_token_count(text: str, token_count: int) -> int:
    """The character offset in text reached after consuming exactly
    token_count tokens (clamped to len(text) if token_count exceeds the
    text's total token count). Exact, not approximated: tiktoken's encode
    partitions text into tokens whose decoded forms concatenate back to
    text exactly, so decoding any whole-token prefix recovers an exact
    prefix of the original text."""
    tokens = _ENCODING.encode(text)
    token_count = max(0, min(token_count, len(tokens)))
    return len(_ENCODING.decode(tokens[:token_count]))


def _snap_to_paragraph_break(text: str, approx_char_offset: int, tolerance_tokens: int) -> int:
    """Search for a blank-line (paragraph) boundary nearest approx_char_offset,
    within an approximate character radius derived from tolerance_tokens.
    The radius is a heuristic search bound only -- if no boundary is found
    within it, approx_char_offset is returned unchanged (no snap), never
    an inexact substitute for the original offset."""
    radius_chars = tolerance_tokens * _CHARS_PER_TOKEN_ESTIMATE
    window_start = max(0, approx_char_offset - radius_chars)
    window_end = min(len(text), approx_char_offset + radius_chars)

    best_offset = None
    best_distance = None
    for match in re.finditer(r"\n[ \t]*\n", text[window_start:window_end]):
        candidate = window_start + match.end()
        distance = abs(candidate - approx_char_offset)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_offset = candidate

    return best_offset if best_offset is not None else approx_char_offset


def _compute_window_boundaries(
    span_text: str, max_tokens: int, overlap_tokens: int, snap_tolerance_tokens: int
) -> list[tuple[int, int]]:
    """Pure, deterministic computation of (local_char_start, local_char_end)
    boundaries within span_text, each window holding at most max_tokens
    tokens, consecutive windows overlapping by overlap_tokens, every
    non-final boundary snapped to the nearest paragraph break. Boundaries
    always cover [0, len(span_text)) with no gaps."""
    total_tokens = count_tokens(span_text)
    if total_tokens <= max_tokens:
        return [(0, len(span_text))]

    boundaries: list[tuple[int, int]] = []
    cursor_tokens = 0
    window_index = 0
    while cursor_tokens < total_tokens:
        start_tokens = cursor_tokens if window_index == 0 else max(0, cursor_tokens - overlap_tokens)
        end_tokens = min(total_tokens, start_tokens + max_tokens)

        start_char = _char_offset_for_token_count(span_text, start_tokens)
        if end_tokens < total_tokens:
            raw_end_char = _char_offset_for_token_count(span_text, end_tokens)
            snapped_end_char = _snap_to_paragraph_break(span_text, raw_end_char, snap_tolerance_tokens)
            # The snap can move the boundary forward as well as backward.
            # The max_tokens budget is a hard constraint (extraction calls
            # must fit within it) and always wins over the snap: only keep
            # the snapped boundary if it doesn't push this window over
            # budget, otherwise fall back to the unsnapped boundary.
            if count_tokens(span_text[start_char:snapped_end_char]) <= max_tokens:
                end_char = snapped_end_char
            else:
                end_char = raw_end_char
        else:
            end_char = len(span_text)

        boundaries.append((start_char, end_char))

        if end_tokens >= total_tokens:
            break
        cursor_tokens = end_tokens
        window_index += 1

    return boundaries


def _detect_chapters(text: str) -> list[ChapterSpan]:
    """Try each heading pattern in priority order; use the first pattern
    that yields at least MIN_CHAPTERS headings. Returns [] if none qualify
    -- the caller treats this as "chapters unavailable" and falls back to
    fixed windowing."""
    for pattern in _CHAPTER_HEADING_PATTERNS:
        heading_starts = [match.start() for match in pattern.finditer(text)]
        if len(heading_starts) >= MIN_CHAPTERS:
            return _build_chapter_spans(text, heading_starts)
    return []


def _build_chapter_spans(text: str, heading_starts: list[int]) -> list[ChapterSpan]:
    spans: list[ChapterSpan] = []
    chapter_index = 0

    if heading_starts[0] > 0:
        spans.append(ChapterSpan(chapter_index=chapter_index, char_start=0, char_end=heading_starts[0]))
        chapter_index += 1

    for i, start in enumerate(heading_starts):
        end = heading_starts[i + 1] if i + 1 < len(heading_starts) else len(text)
        spans.append(ChapterSpan(chapter_index=chapter_index, char_start=start, char_end=end))
        chapter_index += 1

    return spans


def _fallback_fixed_windows(text: str) -> list[ExtractionWindow]:
    """Chapters unavailable: segment the whole text into fixed-size,
    overlapping, paragraph-snapped windows. Each window is its own
    "chapter" for numbering purposes (window_index stays None -- there is
    no further sub-window split below the fallback granularity itself,
    since FALLBACK_WINDOW_TOKENS < MAX_EXTRACTION_TOKENS)."""
    boundaries = _compute_window_boundaries(
        text, FALLBACK_WINDOW_TOKENS, WINDOW_OVERLAP_TOKENS, PARAGRAPH_SNAP_TOLERANCE_TOKENS
    )
    return [
        ExtractionWindow(chapter_index=i, window_index=None, char_start=start, char_end=end)
        for i, (start, end) in enumerate(boundaries)
    ]


def _split_chapter_if_needed(text: str, chapter: ChapterSpan) -> list[ExtractionWindow]:
    """A chapter within budget becomes one ExtractionWindow with
    window_index=None. An over-budget chapter is split into overlapping
    sub-windows sharing chapter.chapter_index, window_index=0..N-1."""
    span_text = text[chapter.char_start : chapter.char_end]
    boundaries = _compute_window_boundaries(
        span_text, MAX_EXTRACTION_TOKENS, WINDOW_OVERLAP_TOKENS, PARAGRAPH_SNAP_TOLERANCE_TOKENS
    )

    if len(boundaries) == 1:
        return [
            ExtractionWindow(
                chapter_index=chapter.chapter_index,
                window_index=None,
                char_start=chapter.char_start,
                char_end=chapter.char_end,
            )
        ]

    return [
        ExtractionWindow(
            chapter_index=chapter.chapter_index,
            window_index=window_index,
            char_start=chapter.char_start + local_start,
            char_end=chapter.char_start + local_end,
        )
        for window_index, (local_start, local_end) in enumerate(boundaries)
    ]


def segment_into_chapters(text: str) -> list[ChapterSpan]:
    """Chapter-level segmentation only (no long-chapter splitting). Exposed
    primarily for testing/inspection; segment_into_extraction_windows() is
    the function downstream extraction (Slice 3+) actually consumes."""
    if not text:
        raise ValueError("Cannot segment empty text")

    chapters = _detect_chapters(text)
    if chapters:
        return chapters
    return [
        ChapterSpan(chapter_index=window.chapter_index, char_start=window.char_start, char_end=window.char_end)
        for window in _fallback_fixed_windows(text)
    ]


def segment_into_extraction_windows(text: str) -> list[ExtractionWindow]:
    """Full segmentation: detect chapters (or fall back to fixed windows),
    then split any chapter exceeding MAX_EXTRACTION_TOKENS into overlapping
    sub-windows. The returned windows always cover [0, len(text)) with no
    gaps, ordered by char_start."""
    if not text:
        raise ValueError("Cannot segment empty text")

    chapters = _detect_chapters(text)
    if not chapters:
        return _fallback_fixed_windows(text)

    windows: list[ExtractionWindow] = []
    for chapter in chapters:
        windows.extend(_split_chapter_if_needed(text, chapter))
    return windows
