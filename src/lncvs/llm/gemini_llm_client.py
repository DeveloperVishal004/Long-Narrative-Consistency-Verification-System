"""GeminiLLMClient -- the first concrete implementation of the plain
LLMClient protocol (Phase H1), mirroring GeminiStructuredClient's isolation
discipline (the only other module that imports the google-genai SDK) and
its retry/backoff and thinking-token handling, simplified for plain-text
completion: no response_schema, no JSON parsing of the output (the caller,
e.g. lncvs.reasoning.decomposition.parser, owns interpreting raw text).

LLMClient and StructuredLLMClient are deliberately separate protocols
(see lncvs.llm.structured's docstring); this file does not blur that line
-- it is a second, independent client satisfying LLMClient only.
"""

import logging
import time

from lncvs.llm.base import LLMCompletion
from lncvs.llm.config import LLMConfig

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_INITIAL_BACKOFF_SECONDS = 5.0


class GeminiLLMClient:
    """LLMClient backed by the real Gemini API, plain-text completion.

    api_key defaults to None, which lets the underlying genai.Client()
    fall back to the GEMINI_API_KEY environment variable -- no secret
    material is ever read or logged by this class itself.

    Retries on 429/503 (rate limit / transient server error) with
    exponential backoff, identical to GeminiStructuredClient's discipline
    -- required for a multi-hundred-call decomposition run to survive
    transient provider unavailability, not optional polish.
    """

    def __init__(self, config: LLMConfig, system_prompt: str | None = None, api_key: str | None = None) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._config = config
        self._system_prompt = system_prompt

    def complete(self, prompt: str) -> LLMCompletion:
        from google.genai import errors, types

        generation_config = types.GenerateContentConfig(
            system_instruction=self._system_prompt,
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_tokens,
            # Disabled for the identical reason documented in
            # gemini_structured_client.py: Gemini 2.5's invisible "thinking"
            # tokens are drawn from the same max_output_tokens budget,
            # which can silently truncate a short, simple completion.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        backoff = _INITIAL_BACKOFF_SECONDS
        response = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._config.model_name, contents=prompt, config=generation_config
                )
                break
            except errors.APIError as exc:
                is_retryable = getattr(exc, "code", None) in (429, 503)
                if not is_retryable or attempt == _MAX_RETRIES:
                    raise
                logger.warning(
                    "Gemini transient error %s (attempt %d/%d), backing off %.0fs",
                    getattr(exc, "code", None), attempt + 1, _MAX_RETRIES, backoff,
                )
                time.sleep(backoff)
                backoff *= 2

        if response.text is None:
            finish_reason = getattr(getattr(response, "candidates", [None])[0], "finish_reason", None)
            raise ValueError(
                f"Gemini completion returned no content (finish_reason={finish_reason!r}, "
                f"possibly truncated by max_output_tokens={self._config.max_tokens} -- raise it)."
            )

        return LLMCompletion(text=response.text, model_fingerprint=self._config.fingerprint())
