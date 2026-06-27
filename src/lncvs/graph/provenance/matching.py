"""Tiered, deterministic quote-to-span resolution (frozen G2 spec §3).

Tier 1 -- exact canonical match: canon(quote) as a substring of
canon(window_text). Tier 2 -- bounded fuzzy match, only attempted if Tier
1 fails: token-level alignment, accepted only if the best contiguous span
has token-overlap >= fuzzy_overlap_threshold AND is unique (no second span
within fuzzy_uniqueness_margin). Tier 3 -- failure: no match recorded.

No LLM calls happen here. Deterministic given identical input.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from lncvs.graph.provenance.canon import canonicalize_with_offsets
from lncvs.graph.provenance.config import ProvenanceConfig


class MatchTier(str, Enum):
    """Which tier resolved (or failed to resolve) a single evidence quote."""

    EXACT = "EXACT"
    FUZZY = "FUZZY"
    FAILED = "FAILED"


class QuoteMatch(BaseModel):
    """The resolution outcome for a single evidence quote against a single
    window's text. char_start/char_end are in the *window's local*
    coordinate space (relative to the window_text passed to resolve_quote)
    when tier is not FAILED; None when tier is FAILED. ambiguous is True
    only for an EXACT match that occurred more than once in the window
    (the first occurrence by ascending offset is always the one used) --
    a FUZZY match that wasn't unique is rejected outright (becomes FAILED),
    never returned as an ambiguous FUZZY match.
    """

    model_config = ConfigDict(frozen=True)

    quote: str = Field(..., min_length=1)
    tier: MatchTier
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    ambiguous: bool = Field(default=False)


def _tokenize_with_spans(canon_text: str) -> list[tuple[str, int, int]]:
    """Split canon_text on its single-space separators (canon_text has
    already collapsed all whitespace runs to one space), returning
    (token, char_start, char_end) for each token in order."""
    tokens: list[tuple[str, int, int]] = []
    start: int | None = None
    sentinel_text = canon_text + " "
    for i, ch in enumerate(sentinel_text):
        if ch == " ":
            if start is not None:
                tokens.append((canon_text[start:i], start, i))
                start = None
        elif start is None:
            start = i
    return tokens


def _fuzzy_match(
    quote_tokens: list[str], window_tokens: list[tuple[str, int, int]], threshold: float, margin: float
) -> tuple[int, int] | None:
    """Sliding fixed-length window of len(quote_tokens) tokens; score = the
    fraction of positions where the window token exactly equals the
    corresponding quote token. Returns the canon-text (char_start, char_end)
    of the best-scoring window if it clears threshold AND no other window
    scores within margin of it; otherwise None."""
    n_quote = len(quote_tokens)
    n_window = len(window_tokens)
    if n_quote == 0 or n_quote > n_window:
        return None

    scored: list[tuple[float, int]] = []
    for start_index in range(n_window - n_quote + 1):
        matches = sum(
            1 for offset in range(n_quote) if window_tokens[start_index + offset][0] == quote_tokens[offset]
        )
        scored.append((matches / n_quote, start_index))

    scored.sort(key=lambda item: -item[0])
    best_score, best_start = scored[0]
    if best_score < threshold:
        return None
    if len(scored) > 1:
        second_score = scored[1][0]
        if best_score - second_score < margin:
            return None

    char_start = window_tokens[best_start][1]
    char_end = window_tokens[best_start + n_quote - 1][2]
    return char_start, char_end


def resolve_quote(quote: str, window_text: str, config: ProvenanceConfig | None = None) -> QuoteMatch:
    """Resolve a single evidence quote against window_text, in window-local
    character coordinates."""
    config = config or ProvenanceConfig()
    canon_window, window_offsets = canonicalize_with_offsets(window_text)
    canon_quote, _ = canonicalize_with_offsets(quote)

    if not canon_quote:
        return QuoteMatch(quote=quote, tier=MatchTier.FAILED)

    occurrences = canon_window.count(canon_quote)
    if occurrences >= 1:
        idx = canon_window.find(canon_quote)
        raw_start = window_offsets[idx]
        raw_end = window_offsets[idx + len(canon_quote) - 1] + 1
        return QuoteMatch(
            quote=quote, tier=MatchTier.EXACT, char_start=raw_start, char_end=raw_end, ambiguous=occurrences > 1
        )

    window_tokens = _tokenize_with_spans(canon_window)
    quote_tokens = [token for token, _, _ in _tokenize_with_spans(canon_quote)]
    fuzzy_span = _fuzzy_match(quote_tokens, window_tokens, config.fuzzy_overlap_threshold, config.fuzzy_uniqueness_margin)
    if fuzzy_span is None:
        return QuoteMatch(quote=quote, tier=MatchTier.FAILED)

    canon_start, canon_end = fuzzy_span
    raw_start = window_offsets[canon_start]
    raw_end = window_offsets[canon_end - 1] + 1
    return QuoteMatch(quote=quote, tier=MatchTier.FUZZY, char_start=raw_start, char_end=raw_end)
