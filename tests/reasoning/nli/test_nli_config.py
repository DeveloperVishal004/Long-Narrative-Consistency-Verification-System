"""NLIConfig validation and fingerprint tests."""

import pytest
from pydantic import ValidationError

from lncvs.reasoning.nli import NLIConfig


def test_nli_config_defaults() -> None:
    config = NLIConfig(model_name="cross-encoder/nli-deberta-v3-base")
    assert config.max_length == 256
    assert config.device == "cpu"


def test_nli_config_rejects_empty_model_name() -> None:
    with pytest.raises(ValidationError):
        NLIConfig(model_name="")


def test_nli_config_rejects_non_positive_max_length() -> None:
    with pytest.raises(ValidationError):
        NLIConfig(model_name="some-model", max_length=0)


def test_fingerprint_is_stable_across_identical_configs() -> None:
    config_a = NLIConfig(model_name="some-model", max_length=128, device="cpu")
    config_b = NLIConfig(model_name="some-model", max_length=128, device="cpu")
    assert config_a.fingerprint() == config_b.fingerprint()


def test_fingerprint_differs_for_different_model_name() -> None:
    config_a = NLIConfig(model_name="model-a")
    config_b = NLIConfig(model_name="model-b")
    assert config_a.fingerprint() != config_b.fingerprint()


def test_fingerprint_differs_for_different_max_length() -> None:
    config_a = NLIConfig(model_name="some-model", max_length=128)
    config_b = NLIConfig(model_name="some-model", max_length=256)
    assert config_a.fingerprint() != config_b.fingerprint()
