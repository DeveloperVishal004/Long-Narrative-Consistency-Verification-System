"""GeminiLLMClient: protocol conformance and error handling, tested without
any real API call. Mirrors test_gemini_structured_client.py exactly, minus
the schema-translation tests (GeminiLLMClient has no response_schema)."""

from unittest.mock import MagicMock, patch

import pytest

from lncvs.llm import GeminiLLMClient, LLMClient, LLMConfig


def test_gemini_llm_client_satisfies_protocol() -> None:
    client = GeminiLLMClient(config=LLMConfig(model_name="gemini-2.5-flash"), api_key="fake-key-not-used")
    assert isinstance(client, LLMClient)


def test_returns_completion_text_and_fingerprint() -> None:
    client = GeminiLLMClient(config=LLMConfig(model_name="gemini-2.5-flash"), api_key="fake-key-not-used")
    fake_response = MagicMock()
    fake_response.text = '["John lost his left arm in an accident."]'
    client._client = MagicMock()
    client._client.models.generate_content.return_value = fake_response

    completion = client.complete("decompose: ...")

    assert completion.text == '["John lost his left arm in an accident."]'
    assert completion.model_fingerprint == LLMConfig(model_name="gemini-2.5-flash").fingerprint()


def test_no_content_response_raises_actionable_value_error() -> None:
    client = GeminiLLMClient(config=LLMConfig(model_name="gemini-2.5-flash", max_tokens=64), api_key="fake-key-not-used")
    fake_candidate = MagicMock()
    fake_candidate.finish_reason = "MAX_TOKENS"
    fake_response = MagicMock()
    fake_response.text = None
    fake_response.candidates = [fake_candidate]
    client._client = MagicMock()
    client._client.models.generate_content.return_value = fake_response

    with pytest.raises(ValueError) as exc_info:
        client.complete("decompose: ...")

    message = str(exc_info.value)
    assert "max_output_tokens" in message
    assert "64" in message


def test_retries_on_503_server_error_then_succeeds() -> None:
    from google.genai import errors

    client = GeminiLLMClient(config=LLMConfig(model_name="gemini-2.5-flash"), api_key="fake-key-not-used")
    client._client = MagicMock()

    server_error = errors.ServerError(code=503, response_json={"error": {"message": "high demand"}})
    success_response = MagicMock()
    success_response.text = "[]"
    success_response.candidates = []

    client._client.models.generate_content.side_effect = [server_error, success_response]

    with patch("time.sleep"):
        completion = client.complete("decompose: ...")

    assert completion.text == "[]"
    assert client._client.models.generate_content.call_count == 2


def test_does_not_retry_on_non_retryable_client_error() -> None:
    from google.genai import errors

    client = GeminiLLMClient(config=LLMConfig(model_name="gemini-2.5-flash"), api_key="fake-key-not-used")
    client._client = MagicMock()
    bad_request = errors.ClientError(code=400, response_json={"error": {"message": "bad request"}})
    client._client.models.generate_content.side_effect = bad_request

    with pytest.raises(errors.ClientError):
        client.complete("decompose: ...")

    assert client._client.models.generate_content.call_count == 1
