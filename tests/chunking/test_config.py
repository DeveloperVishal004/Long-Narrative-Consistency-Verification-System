"""ChunkingConfig validation tests."""

import pytest
from pydantic import ValidationError

from lncvs.chunking import ChunkingConfig


def test_valid_config_constructs() -> None:
    config = ChunkingConfig(chunk_size=200, overlap=20)
    assert config.chunk_size == 200


def test_overlap_equal_to_chunk_size_rejected() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=200, overlap=200)


def test_overlap_greater_than_chunk_size_rejected() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=200, overlap=250)


def test_non_positive_chunk_size_rejected() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=0, overlap=0)
