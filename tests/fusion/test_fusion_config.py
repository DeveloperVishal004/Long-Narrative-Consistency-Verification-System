"""FusionConfig validation and fingerprint tests."""

import pytest
from pydantic import ValidationError

from lncvs.fusion import FusionConfig


def test_fusion_config_defaults() -> None:
    config = FusionConfig()
    assert config.rrf_k == 60
    assert config.top_k_fused == 10


def test_fusion_config_rejects_non_positive_rrf_k() -> None:
    with pytest.raises(ValidationError):
        FusionConfig(rrf_k=0)


def test_fusion_config_rejects_non_positive_top_k_fused() -> None:
    with pytest.raises(ValidationError):
        FusionConfig(top_k_fused=0)


def test_fingerprint_is_stable_across_identical_configs() -> None:
    config_a = FusionConfig(rrf_k=60, top_k_fused=10)
    config_b = FusionConfig(rrf_k=60, top_k_fused=10)
    assert config_a.fingerprint() == config_b.fingerprint()


def test_fingerprint_differs_for_different_rrf_k() -> None:
    config_a = FusionConfig(rrf_k=60)
    config_b = FusionConfig(rrf_k=30)
    assert config_a.fingerprint() != config_b.fingerprint()


def test_fingerprint_differs_for_different_top_k_fused() -> None:
    config_a = FusionConfig(top_k_fused=10)
    config_b = FusionConfig(top_k_fused=5)
    assert config_a.fingerprint() != config_b.fingerprint()
