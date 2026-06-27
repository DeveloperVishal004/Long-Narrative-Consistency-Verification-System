"""Narrative ingestion: loading raw text and cleaning it into a RawDocument."""

from lncvs.ingestion.cleaning import clean_text
from lncvs.ingestion.loader import load_text_file
from lncvs.ingestion.service import load_and_clean_narrative

__all__ = ["clean_text", "load_and_clean_narrative", "load_text_file"]
