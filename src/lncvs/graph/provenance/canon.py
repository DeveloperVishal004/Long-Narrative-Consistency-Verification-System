"""Deterministic text canonicalization with an exact offset map back to
the original string -- the single shared transform both sides of every
quote match go through (frozen G2 spec §3): "Both sides of every match go
through the same canon(), so match offsets are recoverable to true
cleaned-text offsets."

Transform: smart quotes/dashes/ellipsis -> ASCII, whitespace runs
collapsed to a single space, leading/trailing whitespace stripped.

Deliberately does NOT apply Unicode NFC normalization. NFC can change a
string's length (composing/decomposing combining characters), which would
require tracking a many-to-many offset map rather than the simple
one-output-char-to-one-input-index map this module provides. For the two
real project novels (English prose, already NFC-normalized as a practical
matter), this is an empirically safe scope limitation, not a silent gap:
if a real NFC mismatch ever caused a quote to fail matching, the result is
a Tier-3 failure routed to quarantine (see matching.py) -- never a wrong
match. This is a disclosed, deliberate simplification, not an oversight.
"""

_SMART_CHAR_MAP = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
}


def canonicalize_with_offsets(text: str) -> tuple[str, list[int]]:
    """Return (canon_text, offsets) where offsets[i] is the index in the
    original text that produced canon_text[i]. canon_text never exceeds
    len(text) - 1 entries in offsets is always true; every offsets entry
    is a valid index into the original text (never an end-exclusive
    sentinel), since callers reconstruct an end offset as offsets[k]+1.
    """
    output_chars: list[str] = []
    offsets: list[int] = []
    space_run_start: int | None = None

    for i, ch in enumerate(text):
        if ch.isspace():
            if space_run_start is None:
                space_run_start = i
            continue

        if space_run_start is not None:
            if output_chars:
                output_chars.append(" ")
                offsets.append(space_run_start)
            space_run_start = None

        mapped = _SMART_CHAR_MAP.get(ch, ch)
        for mapped_char in mapped:
            output_chars.append(mapped_char)
            offsets.append(i)

    canon_text = "".join(output_chars)

    start = 0
    end = len(canon_text)
    while start < end and canon_text[start] == " ":
        start += 1
    while end > start and canon_text[end - 1] == " ":
        end -= 1

    return canon_text[start:end], offsets[start:end]
