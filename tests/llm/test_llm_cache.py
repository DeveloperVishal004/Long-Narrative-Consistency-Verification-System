"""LLMCache / CachingLLMClient tests."""

from pathlib import Path

from lncvs.llm import CachingLLMClient, InMemoryLLMCache, JsonlLLMCache, LLMCache, LLMClient, LLMCompletion, LLMConfig
from tests.llm.fakes import FakeLLMClient


def _config(model_name: str = "fake-model-a") -> LLMConfig:
    return LLMConfig(model_name=model_name)


def test_in_memory_cache_round_trip() -> None:
    cache = InMemoryLLMCache()
    assert cache.get("missing-key") is None

    completion = LLMCompletion(text="hello", model_fingerprint="fake-model")
    cache.put("key", completion)
    assert cache.get("key") == completion


def test_in_memory_cache_satisfies_llm_cache_protocol() -> None:
    assert isinstance(InMemoryLLMCache(), LLMCache)


def test_fake_llm_client_satisfies_llm_client_protocol() -> None:
    assert isinstance(FakeLLMClient(default_response="[]"), LLMClient)


def test_caching_llm_client_satisfies_llm_client_protocol() -> None:
    caching = CachingLLMClient(FakeLLMClient(default_response="[]"), InMemoryLLMCache(), _config())
    assert isinstance(caching, LLMClient)


def test_caching_client_returns_identical_completion_to_wrapped_client() -> None:
    fake = FakeLLMClient(default_response='["John played piano"]')
    caching = CachingLLMClient(fake, InMemoryLLMCache(), _config())

    direct = fake.complete("decompose: John played piano")
    cached = caching.complete("decompose: John played piano")

    assert direct == cached


def test_second_identical_call_is_served_from_cache() -> None:
    fake = FakeLLMClient(default_response='["John played piano"]')
    caching = CachingLLMClient(fake, InMemoryLLMCache(), _config())

    caching.complete("decompose: John played piano")
    caching.complete("decompose: John played piano")

    assert len(fake.calls) == 1


def test_caching_client_caches_the_raw_completion_not_a_parsed_value() -> None:
    fake = FakeLLMClient(default_response='["John played piano", "John used both hands"]')
    cache = InMemoryLLMCache()
    caching = CachingLLMClient(fake, cache, _config())

    result = caching.complete("decompose: John played a two-handed piano piece")

    assert result.text == '["John played piano", "John used both hands"]'


def test_different_config_fingerprints_do_not_cross_serve_completions() -> None:
    shared_cache = InMemoryLLMCache()
    fake_a = FakeLLMClient(default_response="response-a")
    fake_b = FakeLLMClient(default_response="response-b")

    caching_a = CachingLLMClient(fake_a, shared_cache, _config("model-a"))
    caching_b = CachingLLMClient(fake_b, shared_cache, _config("model-b"))

    result_a = caching_a.complete("shared prompt")
    result_b = caching_b.complete("shared prompt")

    assert len(fake_a.calls) == 1
    assert len(fake_b.calls) == 1
    assert result_a.text == "response-a"
    assert result_b.text == "response-b"


def test_prompt_change_naturally_invalidates_the_cache_key() -> None:
    """A different rendered prompt (e.g. from a template edit) must not hit a
    cache entry from a different prompt, even under the same config."""
    fake = FakeLLMClient(scripted={"prompt-v1": "result-v1", "prompt-v2": "result-v2"})
    caching = CachingLLMClient(fake, InMemoryLLMCache(), _config())

    result_v1 = caching.complete("prompt-v1")
    result_v2 = caching.complete("prompt-v2")

    assert result_v1.text == "result-v1"
    assert result_v2.text == "result-v2"
    assert len(fake.calls) == 2


def test_jsonl_cache_satisfies_llm_cache_protocol(tmp_path: Path) -> None:
    assert isinstance(JsonlLLMCache(tmp_path / "decomposition_cache.jsonl"), LLMCache)


def test_jsonl_cache_persists_across_independent_instances(tmp_path: Path) -> None:
    cache_path = tmp_path / "decomposition_cache.jsonl"

    fake = FakeLLMClient(default_response='["John lost his left arm in 2010."]')
    first_instance = JsonlLLMCache(cache_path)
    caching_first = CachingLLMClient(fake, first_instance, _config())
    caching_first.complete("decompose: John lost his left arm in 2010.")

    assert cache_path.exists()

    second_instance = JsonlLLMCache(cache_path)
    fake_unused = FakeLLMClient()  # should never be called -- cache hit
    caching_second = CachingLLMClient(fake_unused, second_instance, _config())
    result = caching_second.complete("decompose: John lost his left arm in 2010.")

    assert result.text == '["John lost his left arm in 2010."]'
    assert fake_unused.calls == []


def test_jsonl_cache_survives_a_crash_between_writes(tmp_path: Path) -> None:
    """Per-write append (not batched) means completions already paid for
    before a crash are never lost."""
    cache_path = tmp_path / "decomposition_cache.jsonl"
    fake = FakeLLMClient(scripted={"claim-1": '["fact-1"]', "claim-2": '["fact-2"]'})
    caching = CachingLLMClient(fake, JsonlLLMCache(cache_path), _config())

    caching.complete("claim-1")
    # Simulate a crash here: only claim-1 has been written.
    line_count_after_first_write = sum(1 for _ in cache_path.open())
    assert line_count_after_first_write == 1

    reloaded = JsonlLLMCache(cache_path)
    assert reloaded.get(caching._cache_key("claim-1")) is not None
    assert reloaded.get(caching._cache_key("claim-2")) is None
