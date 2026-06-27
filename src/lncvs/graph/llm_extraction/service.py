"""Window extraction service: ties an injected StructuredLLMClient to a
single extraction window and returns a validated WindowExtraction.

Mirrors lncvs.reasoning.decomposition.service's shape exactly: a small
Protocol (WindowExtractor), one LLM-backed implementation
(LLMWindowExtractor) holding no state beyond its injected dependencies,
and a pure parser (parse_window_extraction) that is fully unit-testable
offline without a client, identical in spirit to parse_decomposition_response.
"""

from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from lncvs.graph.llm_extraction.config import ExtractionConfig
from lncvs.graph.llm_extraction.json_schema import EXTRACTION_JSON_SCHEMA
from lncvs.graph.llm_extraction.prompts import render_user_prompt
from lncvs.graph.llm_extraction.schema import WindowExtraction
from lncvs.llm import StructuredLLMClient


@runtime_checkable
class WindowExtractor(Protocol):
    """Contract for extracting entities/relations/events from a single window."""

    def extract(self, window_text: str, chapter_index: int, window_index: int | None) -> WindowExtraction:
        """Return the validated extraction result for window_text."""
        ...


class LLMWindowExtractor:
    """WindowExtractor backed by an injected StructuredLLMClient.

    Holds no state beyond its injected dependencies. Determinism rests
    entirely on the StructuredLLMClient (wrap in CachingStructuredLLMClient
    for real provider calls; FakeStructuredLLMClient is deterministic by
    construction for tests).
    """

    def __init__(self, client: StructuredLLMClient, config: ExtractionConfig | None = None) -> None:
        self._client = client
        self._config = config or ExtractionConfig()

    def extract(self, window_text: str, chapter_index: int, window_index: int | None) -> WindowExtraction:
        if not window_text or not window_text.strip():
            raise ValueError("window_text must not be empty")

        prompt = render_user_prompt(window_text, chapter_index, window_index)
        completion = self._client.complete_structured(prompt, EXTRACTION_JSON_SCHEMA)
        return parse_window_extraction(completion.data)


def parse_window_extraction(raw: dict) -> WindowExtraction:
    """Pure parser: raw structured-completion dict -> WindowExtraction.

    No LLM calls happen here -- deterministic given identical input. Never
    lets a bare pydantic.ValidationError escape this module: any shape
    violation (a malformed local_id, an out-of-vocabulary enum value the
    provider's strict mode should have prevented but this layer verifies
    independently, a missing evidence_quotes entry) is re-raised as a
    plain ValueError with the validation detail attached.
    """
    try:
        return WindowExtraction.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Extraction response failed schema validation: {exc}") from exc
