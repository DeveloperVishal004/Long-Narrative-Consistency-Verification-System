"""RetrievalConfig validation tests."""

import pytest
from pydantic import ValidationError

from lncvs.retrieval import RetrievalConfig


def test_default_top_k() -> None:
    config = RetrievalConfig()
    assert config.top_k == 5


def test_custom_top_k() -> None:
    config = RetrievalConfig(top_k=10)
    assert config.top_k == 10


def test_non_positive_top_k_rejected() -> None:
    with pytest.raises(ValidationError):
        RetrievalConfig(top_k=0)
