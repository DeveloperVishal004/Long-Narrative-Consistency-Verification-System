"""LLMClaimDecomposer tests — offline via FakeLLMClient."""

import pytest

from lncvs.llm import CachingLLMClient, InMemoryLLMCache, LLMConfig
from lncvs.reasoning.decomposition import ClaimDecomposer, DecompositionConfig, LLMClaimDecomposer, make_source_claim_id
from tests.llm.fakes import FakeLLMClient

DUMMY_CLAIM = "John played a two-handed piano piece in London."
DUMMY_RESPONSE = '["John played piano", "John used both hands", "the event occurred in London"]'


def _config(max_atomic_claims: int = 10) -> DecompositionConfig:
    return DecompositionConfig(llm_config=LLMConfig(model_name="fake-model"), max_atomic_claims=max_atomic_claims)


def test_decomposer_satisfies_claim_decomposer_protocol() -> None:
    decomposer = LLMClaimDecomposer(FakeLLMClient(default_response=DUMMY_RESPONSE), _config())
    assert isinstance(decomposer, ClaimDecomposer)


def test_decompose_dummy_claim_returns_expected_atomic_claims() -> None:
    decomposer = LLMClaimDecomposer(FakeLLMClient(default_response=DUMMY_RESPONSE), _config())

    claims = decomposer.decompose(DUMMY_CLAIM)

    assert [c.text for c in claims] == [
        "John played piano",
        "John used both hands",
        "the event occurred in London",
    ]


def test_decompose_sets_parent_claim_id_to_the_source_claim_id() -> None:
    decomposer = LLMClaimDecomposer(FakeLLMClient(default_response=DUMMY_RESPONSE), _config())

    claims = decomposer.decompose(DUMMY_CLAIM)

    expected_parent_id = make_source_claim_id(DUMMY_CLAIM)
    assert all(c.parent_claim_id == expected_parent_id for c in claims)


def test_decompose_is_deterministic_across_fresh_decomposer_instances() -> None:
    first_decomposer = LLMClaimDecomposer(FakeLLMClient(default_response=DUMMY_RESPONSE), _config())
    second_decomposer = LLMClaimDecomposer(FakeLLMClient(default_response=DUMMY_RESPONSE), _config())

    first_claims = first_decomposer.decompose(DUMMY_CLAIM)
    second_claims = second_decomposer.decompose(DUMMY_CLAIM)

    assert [c.claim_id for c in first_claims] == [c.claim_id for c in second_claims]


def test_decompose_with_caching_llm_client_invokes_wrapped_client_once_on_repeat() -> None:
    fake = FakeLLMClient(default_response=DUMMY_RESPONSE)
    config = _config()
    caching_client = CachingLLMClient(fake, InMemoryLLMCache(), config.llm_config)
    decomposer = LLMClaimDecomposer(caching_client, config)

    decomposer.decompose(DUMMY_CLAIM)
    decomposer.decompose(DUMMY_CLAIM)

    assert len(fake.calls) == 1


def test_decompose_empty_claim_raises() -> None:
    decomposer = LLMClaimDecomposer(FakeLLMClient(default_response=DUMMY_RESPONSE), _config())
    with pytest.raises(ValueError, match="must not be empty"):
        decomposer.decompose("")


def test_decompose_blank_claim_raises() -> None:
    decomposer = LLMClaimDecomposer(FakeLLMClient(default_response=DUMMY_RESPONSE), _config())
    with pytest.raises(ValueError, match="must not be empty"):
        decomposer.decompose("   ")


def test_decompose_respects_max_atomic_claims_guard() -> None:
    decomposer = LLMClaimDecomposer(FakeLLMClient(default_response=DUMMY_RESPONSE), _config(max_atomic_claims=2))
    with pytest.raises(ValueError, match="max_atomic_claims"):
        decomposer.decompose(DUMMY_CLAIM)


def test_llm_client_error_propagates_without_silent_fallback() -> None:
    """No scripted/default response configured -> FakeLLMClient raises -> must propagate, not return []."""
    decomposer = LLMClaimDecomposer(FakeLLMClient(), _config())
    with pytest.raises(ValueError, match="no scripted response"):
        decomposer.decompose(DUMMY_CLAIM)
