"""StructuredCompletion / StructuredLLMClient protocol conformance and
DTO validation tests."""

import pytest
from pydantic import ValidationError

from lncvs.llm import StructuredCompletion, StructuredLLMClient
from tests.llm.fakes import FakeStructuredLLMClient


def test_fake_structured_llm_client_satisfies_protocol() -> None:
    assert isinstance(FakeStructuredLLMClient(default_response={"entities": []}), StructuredLLMClient)


def test_structured_completion_requires_non_empty_ids() -> None:
    with pytest.raises(ValidationError):
        StructuredCompletion(data={}, model_fingerprint="", response_id="resp-1")
    with pytest.raises(ValidationError):
        StructuredCompletion(data={}, model_fingerprint="fp", response_id="")


def test_structured_completion_is_frozen() -> None:
    completion = StructuredCompletion(data={"a": 1}, model_fingerprint="fp", response_id="resp-1")
    with pytest.raises(ValidationError):
        completion.model_fingerprint = "other"


def test_fake_structured_llm_client_returns_scripted_response_by_prompt() -> None:
    fake = FakeStructuredLLMClient(scripted={"prompt-a": {"x": 1}, "prompt-b": {"x": 2}})

    result_a = fake.complete_structured("prompt-a", response_schema={})
    result_b = fake.complete_structured("prompt-b", response_schema={})

    assert result_a.data == {"x": 1}
    assert result_b.data == {"x": 2}
    assert len(fake.calls) == 2


def test_fake_structured_llm_client_raises_for_unscripted_prompt() -> None:
    fake = FakeStructuredLLMClient(scripted={"known": {"x": 1}})
    with pytest.raises(ValueError):
        fake.complete_structured("unknown", response_schema={})
