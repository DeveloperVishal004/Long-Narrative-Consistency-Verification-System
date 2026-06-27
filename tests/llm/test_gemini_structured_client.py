"""GeminiStructuredClient: protocol conformance and the pure schema-dialect
translator (_to_gemini_schema), tested without any real API call."""

from unittest.mock import MagicMock, patch

import pytest

from lncvs.llm import GeminiStructuredClient, LLMConfig, StructuredLLMClient
from lncvs.llm.gemini_structured_client import _to_gemini_schema


def test_gemini_structured_client_satisfies_protocol() -> None:
    from lncvs.llm import LLMConfig

    client = GeminiStructuredClient(config=LLMConfig(model_name="gemini-2.5-flash"), system_prompt="sys", api_key="fake-key-not-used")
    assert isinstance(client, StructuredLLMClient)


def test_translates_nullable_union_type_to_nullable_flag() -> None:
    schema = {"type": ["string", "null"]}
    translated = _to_gemini_schema(schema)
    assert translated == {"type": "string", "nullable": True}


def test_translates_object_or_null_type() -> None:
    schema = {"type": ["object", "null"], "properties": {}}
    translated = _to_gemini_schema(schema)
    assert translated["type"] == "object"
    assert translated["nullable"] is True


def test_strips_additional_properties() -> None:
    schema = {"type": "object", "additionalProperties": False, "properties": {}}
    translated = _to_gemini_schema(schema)
    assert "additionalProperties" not in translated


def test_recurses_into_nested_structures() -> None:
    schema = {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"temporal": {"type": ["object", "null"]}},
                },
            }
        },
    }
    translated = _to_gemini_schema(schema)
    temporal = translated["properties"]["events"]["items"]["properties"]["temporal"]
    assert temporal == {"type": "object", "nullable": True}
    assert "additionalProperties" not in translated["properties"]["events"]["items"]


def test_leaves_non_nullable_schema_unchanged_in_shape() -> None:
    schema = {"type": "string", "minLength": 1}
    translated = _to_gemini_schema(schema)
    assert translated == {"type": "string", "minLength": 1}


def test_does_not_mutate_the_original_schema() -> None:
    schema = {"type": ["string", "null"]}
    _to_gemini_schema(schema)
    assert schema == {"type": ["string", "null"]}


def test_translation_is_deterministic() -> None:
    from lncvs.graph.llm_extraction import EXTRACTION_JSON_SCHEMA

    assert _to_gemini_schema(EXTRACTION_JSON_SCHEMA) == _to_gemini_schema(EXTRACTION_JSON_SCHEMA)


def test_truncated_response_raises_actionable_value_error_not_bare_json_error() -> None:
    """Pins down the real failure mode hit during the first live run:
    max_output_tokens too small truncated the JSON mid-string. The error
    must name the likely cause and the config value to raise, not just
    propagate a bare JSONDecodeError."""
    client = GeminiStructuredClient(config=LLMConfig(model_name="gemini-2.5-flash", max_tokens=128), system_prompt="sys", api_key="fake-key-not-used")

    fake_candidate = MagicMock()
    fake_candidate.finish_reason = "MAX_TOKENS"
    fake_response = MagicMock()
    fake_response.text = '{"entities": [{"local_id": "e1", "name": "Jo'  # truncated mid-string
    fake_response.candidates = [fake_candidate]
    client._client = MagicMock()
    client._client.models.generate_content.return_value = fake_response

    with pytest.raises(ValueError) as exc_info:
        client.complete_structured("prompt", {"type": "object"})

    message = str(exc_info.value)
    assert "max_output_tokens" in message
    assert "128" in message


def test_retries_on_503_server_error_then_succeeds() -> None:
    """Pins down the second real failure mode hit live: a 503 ServerError
    is a different exception class than the 429 ClientError the retry
    loop originally only caught, so it crashed the whole run uncaught on
    the very first transient "high demand" response. Both must retry."""
    from google.genai import errors

    client = GeminiStructuredClient(config=LLMConfig(model_name="gemini-2.5-flash"), system_prompt="sys", api_key="fake-key-not-used")
    client._client = MagicMock()

    server_error = errors.ServerError(code=503, response_json={"error": {"message": "high demand"}})
    success_response = MagicMock()
    success_response.text = '{"ok": true}'
    success_response.candidates = []
    success_response.response_id = "resp-1"

    client._client.models.generate_content.side_effect = [server_error, success_response]

    with patch("time.sleep"):
        completion = client.complete_structured("prompt", {"type": "object"})

    assert completion.data == {"ok": True}
    assert client._client.models.generate_content.call_count == 2


def test_does_not_retry_on_non_retryable_client_error() -> None:
    from google.genai import errors

    client = GeminiStructuredClient(config=LLMConfig(model_name="gemini-2.5-flash"), system_prompt="sys", api_key="fake-key-not-used")
    client._client = MagicMock()
    bad_request = errors.ClientError(code=400, response_json={"error": {"message": "bad request"}})
    client._client.models.generate_content.side_effect = bad_request

    with pytest.raises(errors.ClientError):
        client.complete_structured("prompt", {"type": "object"})

    assert client._client.models.generate_content.call_count == 1
