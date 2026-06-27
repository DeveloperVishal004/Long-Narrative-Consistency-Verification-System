"""DocumentChunk validation tests."""

import pytest
from pydantic import ValidationError

from lncvs.schemas import DocumentChunk


def _build_chunk(**overrides: object) -> DocumentChunk:
    defaults: dict = {
        "chunk_id": "chunk-0001",
        "text": "John lost his left arm in an accident in 2010.",
        "char_start": 0,
        "char_end": 48,
        "chapter": None,
        "source_id": "john_test",
    }
    defaults.update(overrides)
    return DocumentChunk(**defaults)


def test_valid_chunk_constructs() -> None:
    chunk = _build_chunk()
    assert chunk.chunk_id == "chunk-0001"
    assert chunk.char_end > chunk.char_start


def test_char_end_must_exceed_char_start() -> None:
    with pytest.raises(ValidationError):
        _build_chunk(char_start=10, char_end=10)


def test_char_end_less_than_char_start_rejected() -> None:
    with pytest.raises(ValidationError):
        _build_chunk(char_start=10, char_end=5)


def test_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        _build_chunk(text="")


def test_empty_chunk_id_rejected() -> None:
    with pytest.raises(ValidationError):
        _build_chunk(chunk_id="")
