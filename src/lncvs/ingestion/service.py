"""Ingestion service: composes loading and cleaning into a typed RawDocument."""

import logging
from pathlib import Path

from lncvs.ingestion.cleaning import clean_text
from lncvs.ingestion.loader import load_text_file
from lncvs.schemas import RawDocument

logger = logging.getLogger(__name__)


def load_and_clean_narrative(path: Path, source_id: str) -> RawDocument:
    """Load a narrative file from disk and return its cleaned, typed representation."""
    raw_text = load_text_file(path)
    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        raise ValueError(f"Narrative file {path} produced empty content after cleaning")

    logger.info("Ingested narrative %r from %s", source_id, path)
    return RawDocument(source_id=source_id, raw_text=raw_text, cleaned_text=cleaned_text)
