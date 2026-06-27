"""Narrative loading from disk."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_text_file(path: Path) -> str:
    """Load a narrative text file as a single string.

    Raises:
        FileNotFoundError: if path does not exist, with an actionable message.
        ValueError: if the file cannot be decoded as UTF-8.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Narrative file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        logger.error("Failed to decode %s as UTF-8: %s", path, exc)
        raise ValueError(f"Narrative file {path} is not valid UTF-8 text") from exc

    logger.info("Loaded narrative file %s (%d characters)", path, len(text))
    return text
