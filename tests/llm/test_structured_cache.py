"""StructuredLLMCache / CachingStructuredLLMClient / JsonlStructuredLLMCache
tests -- mirrors test_llm_cache.py, extended with schema_version isolation
and file-backed persistence (the Tier-1 reproducibility mechanism)."""

from pathlib import Path

from lncvs.llm import (
    CachingStructuredLLMClient,
    InMemoryStructuredLLMCache,
    JsonlStructuredLLMCache,
    LLMConfig,
    StructuredLLMCache,
    StructuredLLMClient,
)
from tests.llm.fakes import FakeStructuredLLMClient


def _config(model_name: str = "fake-model-a") -> LLMConfig:
    return LLMConfig(model_name=model_name)


def test_in_memory_structured_cache_round_trip() -> None:
    cache = InMemoryStructuredLLMCache()
    assert cache.get("missing-key") is None

    fake = FakeStructuredLLMClient(default_response={"x": 1})
    completion = fake.complete_structured("prompt", {})
    cache.put("key", completion)
    assert cache.get("key") == completion


def test_in_memory_structured_cache_satisfies_protocol() -> None:
    assert isinstance(InMemoryStructuredLLMCache(), StructuredLLMCache)


def test_caching_structured_client_satisfies_protocol() -> None:
    caching = CachingStructuredLLMClient(
        FakeStructuredLLMClient(default_response={}), InMemoryStructuredLLMCache(), _config(), schema_version="v1"
    )
    assert isinstance(caching, StructuredLLMClient)


def test_second_identical_call_is_served_from_cache() -> None:
    fake = FakeStructuredLLMClient(default_response={"entities": []})
    caching = CachingStructuredLLMClient(fake, InMemoryStructuredLLMCache(), _config(), schema_version="v1")

    caching.complete_structured("window text", {})
    caching.complete_structured("window text", {})

    assert len(fake.calls) == 1


def test_different_schema_versions_do_not_cross_serve_completions() -> None:
    shared_cache = InMemoryStructuredLLMCache()
    fake_v1 = FakeStructuredLLMClient(default_response={"version": 1})
    fake_v2 = FakeStructuredLLMClient(default_response={"version": 2})

    caching_v1 = CachingStructuredLLMClient(fake_v1, shared_cache, _config(), schema_version="v1")
    caching_v2 = CachingStructuredLLMClient(fake_v2, shared_cache, _config(), schema_version="v2")

    result_v1 = caching_v1.complete_structured("shared prompt", {})
    result_v2 = caching_v2.complete_structured("shared prompt", {})

    assert len(fake_v1.calls) == 1
    assert len(fake_v2.calls) == 1
    assert result_v1.data == {"version": 1}
    assert result_v2.data == {"version": 2}


def test_jsonl_cache_persists_across_independent_instances(tmp_path: Path) -> None:
    cache_path = tmp_path / "extraction_artifacts.jsonl"

    fake = FakeStructuredLLMClient(default_response={"entities": ["e1"]})
    first_instance = JsonlStructuredLLMCache(cache_path)
    caching_first = CachingStructuredLLMClient(fake, first_instance, _config(), schema_version="v1")
    caching_first.complete_structured("window-1", {})

    assert cache_path.exists()

    second_instance = JsonlStructuredLLMCache(cache_path)
    fake_unused = FakeStructuredLLMClient()  # should never be called -- cache hit
    caching_second = CachingStructuredLLMClient(fake_unused, second_instance, _config(), schema_version="v1")
    result = caching_second.complete_structured("window-1", {})

    assert result.data == {"entities": ["e1"]}
    assert fake_unused.calls == []


def test_jsonl_cache_survives_a_crash_between_writes(tmp_path: Path) -> None:
    """Per-write append (not batched) means completions already paid for
    before a crash are never lost."""
    cache_path = tmp_path / "extraction_artifacts.jsonl"
    fake = FakeStructuredLLMClient(scripted={"window-1": {"x": 1}, "window-2": {"x": 2}})
    caching = CachingStructuredLLMClient(fake, JsonlStructuredLLMCache(cache_path), _config(), schema_version="v1")

    caching.complete_structured("window-1", {})
    # Simulate a crash here: only window-1 has been written.
    line_count_after_first_write = sum(1 for _ in cache_path.open())
    assert line_count_after_first_write == 1

    reloaded = JsonlStructuredLLMCache(cache_path)
    assert reloaded.get(caching._cache_key("window-1")) is not None
    assert reloaded.get(caching._cache_key("window-2")) is None
