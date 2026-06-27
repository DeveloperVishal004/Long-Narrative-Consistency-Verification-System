"""Narrative text cleaning.

Cleaning operates on the whole document and does not preserve a character-
offset mapping back to the raw text — see RawDocument's docstring. Chunking
always works over the output of clean_text, so internal consistency between
chunk offsets and cleaned_text is maintained regardless.
"""

import re

_BOM = "﻿"
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(raw_text: str) -> str:
    """Normalize a raw narrative string for chunking.

    Responsibilities:
      - strip a leading UTF-8 BOM, if present
      - normalize CRLF/CR line endings to LF
      - strip trailing whitespace from each line
      - collapse 3+ consecutive blank lines down to exactly 2 (preserves
        paragraph breaks without allowing unbounded blank runs)
      - strip leading/trailing whitespace from the document as a whole
    """
    text = raw_text.removeprefix(_BOM)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()
