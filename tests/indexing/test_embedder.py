"""Embedder protocol conformance tests.

The fast, deterministic path uses FakeEmbedder. The real
SentenceTransformerEmbedder is exercised separately in
tests/acceptance/test_phase1_vertical_slice.py, where a model download is
expected and acceptable; here we only confirm it can be constructed without
raising on import, skipping cleanly if the model cannot be loaded (e.g. no
network access in this environment).
"""

import pytest

from lncvs.indexing import Embedder, EmbeddingConfig, SentenceTransformerEmbedder
from tests.indexing.fakes import FakeEmbedder


def test_fake_embedder_satisfies_embedder_protocol() -> None:
    assert isinstance(FakeEmbedder(), Embedder)


def test_fake_embedder_is_deterministic_across_calls() -> None:
    embedder = FakeEmbedder()
    first = embedder.embed_query("John lost his left arm in an accident.")
    second = embedder.embed_query("John lost his left arm in an accident.")
    assert first == second


def test_fake_embedder_rejects_empty_batch() -> None:
    with pytest.raises(ValueError):
        FakeEmbedder().embed_texts([])


def test_sentence_transformer_embedder_loads_or_skips_cleanly() -> None:
    config = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        embedder = SentenceTransformerEmbedder(config)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Could not load embedding model in this environment: {exc}")

    vector = embedder.embed_query("John lost his left arm in an accident.")
    assert isinstance(vector, list)
    assert len(vector) > 0
