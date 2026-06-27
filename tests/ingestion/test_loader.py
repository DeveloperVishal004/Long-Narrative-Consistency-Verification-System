"""Loader tests."""

from pathlib import Path

import pytest

from lncvs.ingestion import load_text_file

SAMPLE_NARRATIVE = Path(__file__).resolve().parents[2] / "data" / "sample_narrative" / "john_test.txt"


def test_load_text_file_reads_sample_narrative() -> None:
    text = load_text_file(SAMPLE_NARRATIVE)
    assert "John lost his left arm in an accident in 2010." in text
    assert "John moved to London in 2012." in text


def test_load_text_file_raises_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError, match="Narrative file not found"):
        load_text_file(missing)
