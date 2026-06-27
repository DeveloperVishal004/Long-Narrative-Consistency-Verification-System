"""Gated, real-API test for OpenAIStructuredClient.

Skips cleanly if OPENAI_API_KEY is not set in this environment, the same
pattern test_phase5_nli_verdict.py uses for real_embedder/real_nli_model.
Marked slow because it makes a real, billed API call -- excluded from the
default test run (see pyproject.toml's addopts), run explicitly with
`pytest -m slow`.
"""

import os

import pytest

from lncvs.llm import LLMConfig, OpenAIStructuredClient

_TRIVIAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


@pytest.mark.slow
def test_openai_structured_client_returns_schema_conformant_response() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set in this environment")

    config = LLMConfig(model_name="gpt-4o-2024-08-06", temperature=0.0, max_tokens=64)
    client = OpenAIStructuredClient(
        config=config,
        system_prompt="You answer with a single word in the 'answer' field.",
        schema_name="trivial_test_schema",
    )

    completion = client.complete_structured("What is 2+2? Answer with the numeral only.", _TRIVIAL_SCHEMA)

    assert set(completion.data.keys()) == {"answer"}
    assert isinstance(completion.data["answer"], str)
    assert completion.response_id
    assert completion.model_fingerprint == config.fingerprint()
