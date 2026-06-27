"""EmbeddingConfig validation tests."""

from lncvs.indexing import EmbeddingConfig


def test_embedding_config_valid_construction() -> None:
    config = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert config.device == "cpu"
    assert config.normalize_embeddings is True


def test_embedding_config_allows_overriding_device_and_normalization() -> None:
    config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda",
        normalize_embeddings=False,
    )
    assert config.device == "cuda"
    assert config.normalize_embeddings is False


def test_fingerprint_is_stable_across_identical_configs() -> None:
    config_a = EmbeddingConfig(model_name="all-MiniLM-L6-v2", device="cpu", normalize_embeddings=True)
    config_b = EmbeddingConfig(model_name="all-MiniLM-L6-v2", device="cpu", normalize_embeddings=True)

    assert config_a.fingerprint() == config_b.fingerprint()


def test_fingerprint_differs_for_different_model_name() -> None:
    config_a = EmbeddingConfig(model_name="all-MiniLM-L6-v2")
    config_b = EmbeddingConfig(model_name="all-mpnet-base-v2")

    assert config_a.fingerprint() != config_b.fingerprint()


def test_fingerprint_differs_for_different_device() -> None:
    config_a = EmbeddingConfig(model_name="all-MiniLM-L6-v2", device="cpu")
    config_b = EmbeddingConfig(model_name="all-MiniLM-L6-v2", device="cuda")

    assert config_a.fingerprint() != config_b.fingerprint()


def test_fingerprint_differs_for_different_normalization() -> None:
    config_a = EmbeddingConfig(model_name="all-MiniLM-L6-v2", normalize_embeddings=True)
    config_b = EmbeddingConfig(model_name="all-MiniLM-L6-v2", normalize_embeddings=False)

    assert config_a.fingerprint() != config_b.fingerprint()
