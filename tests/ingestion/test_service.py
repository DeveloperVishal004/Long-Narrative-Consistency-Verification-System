"""Ingestion service tests."""

from pathlib import Path

import pytest

from lncvs.ingestion import load_and_clean_narrative
from lncvs.schemas import RawDocument

SAMPLE_NARRATIVE = Path(__file__).resolve().parents[2] / "data" / "sample_narrative" / "john_test.txt"


def test_load_and_clean_narrative_returns_raw_document() -> None:
    document = load_and_clean_narrative(SAMPLE_NARRATIVE, source_id="john_test")

    assert isinstance(document, RawDocument)
    assert document.source_id == "john_test"
    assert "John lost his left arm in an accident in 2010." in document.cleaned_text
    assert document.raw_text != ""


def test_load_and_clean_narrative_raises_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        load_and_clean_narrative(missing, source_id="missing")
