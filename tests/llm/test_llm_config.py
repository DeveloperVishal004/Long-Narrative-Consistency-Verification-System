"""LLMConfig validation and fingerprint tests."""

from lncvs.llm import LLMConfig


def test_llm_config_defaults() -> None:
    config = LLMConfig(model_name="claude-opus-4-8")
    assert config.temperature == 0.0
    assert config.max_tokens == 1024


def test_fingerprint_is_stable_across_identical_configs() -> None:
    config_a = LLMConfig(model_name="claude-opus-4-8", temperature=0.0, max_tokens=512)
    config_b = LLMConfig(model_name="claude-opus-4-8", temperature=0.0, max_tokens=512)

    assert config_a.fingerprint() == config_b.fingerprint()


def test_fingerprint_differs_for_different_model_name() -> None:
    config_a = LLMConfig(model_name="claude-opus-4-8")
    config_b = LLMConfig(model_name="claude-sonnet-4-6")

    assert config_a.fingerprint() != config_b.fingerprint()


def test_fingerprint_differs_for_different_temperature() -> None:
    config_a = LLMConfig(model_name="claude-opus-4-8", temperature=0.0)
    config_b = LLMConfig(model_name="claude-opus-4-8", temperature=0.7)

    assert config_a.fingerprint() != config_b.fingerprint()


def test_fingerprint_differs_for_different_max_tokens() -> None:
    config_a = LLMConfig(model_name="claude-opus-4-8", max_tokens=512)
    config_b = LLMConfig(model_name="claude-opus-4-8", max_tokens=1024)

    assert config_a.fingerprint() != config_b.fingerprint()
