"""GeminiStructuredClient -- the only module in this codebase that imports
the google-genai SDK, mirroring the isolation discipline already applied
to openai (openai_structured_client.py), chromadb, rank_bm25, and networkx.

A second StructuredLLMClient implementation alongside OpenAIStructuredClient
-- the protocol was designed provider-agnostic from the start (Decision 1
of the G2 extraction-interface freeze) specifically so a second provider
could be added without touching the protocol, graph.llm_extraction, or
any other consumer. Added when the project's OpenAI account had no usable
billing quota; Gemini's free tier did.

Schema-dialect adapter, confined entirely to this module: Gemini's
structured-output schema is OpenAPI-3.0-flavored, not strict JSON Schema --
it rejects `"type": [X, "null"]` (JSON Schema's nullable-union form, used
by lncvs.graph.llm_extraction.json_schema.EXTRACTION_JSON_SCHEMA for its
nullable temporal fields) and instead expects `"type": X, "nullable": true`.
_to_gemini_schema() translates one to the other on every call; the
canonical EXTRACTION_JSON_SCHEMA constant itself is never modified, so
OpenAIStructuredClient keeps receiving the unmodified, provider-agnostic
schema. additionalProperties is also stripped (Gemini's schema validator
does not recognize it; the strict-shape guarantee it expresses for OpenAI
has no Gemini equivalent at the API layer, so this is a known, accepted
asymmetry between providers' enforcement strength -- see hallucination
prevention tier 3, the deterministic quote-verification ledger/provenance
boundary, which remains the real backstop for both providers regardless.
"""

import logging
import time

from lncvs.llm.config import LLMConfig
from lncvs.llm.structured import StructuredCompletion

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_INITIAL_BACKOFF_SECONDS = 5.0


def _to_gemini_schema(node: object) -> object:
    """Translate a JSON-Schema-dialect node into Gemini's OpenAPI-flavored
    dialect. Pure, recursive, deterministic -- never mutates its input."""
    if isinstance(node, dict):
        translated = {key: _to_gemini_schema(value) for key, value in node.items()}
        type_value = translated.get("type")
        if isinstance(type_value, list):
            non_null_types = [t for t in type_value if t != "null"]
            translated["type"] = non_null_types[0] if non_null_types else "string"
            if "null" in type_value:
                translated["nullable"] = True
        translated.pop("additionalProperties", None)
        return translated
    if isinstance(node, list):
        return [_to_gemini_schema(item) for item in node]
    return node


class GeminiStructuredClient:
    """StructuredLLMClient backed by the real Gemini API.

    api_key defaults to None, which lets the underlying genai.Client()
    fall back to the GEMINI_API_KEY environment variable -- no secret
    material is ever read or logged by this class itself.

    Retries on 429 (rate limit / quota) with exponential backoff, up to
    _MAX_RETRIES times -- required for a multi-hundred-call extraction run
    to survive the free tier's per-minute request limits, not optional
    polish. Any other exception propagates immediately, uncaught.
    """

    def __init__(self, config: LLMConfig, system_prompt: str, api_key: str | None = None) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._config = config
        self._system_prompt = system_prompt

    def complete_structured(self, prompt: str, response_schema: dict) -> StructuredCompletion:
        from google.genai import errors, types

        gemini_schema = _to_gemini_schema(response_schema)
        generation_config = types.GenerateContentConfig(
            system_instruction=self._system_prompt,
            response_mime_type="application/json",
            response_schema=gemini_schema,
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_tokens,
            # Gemini 2.5's "thinking" tokens are drawn from the SAME
            # max_output_tokens budget as the visible response, invisibly --
            # this caused a real truncation (finish_reason=MAX_TOKENS at
            # only ~5.6KB of a 16384-token budget) during a live run, with
            # no visible content to explain where the budget went.
            # Structured extraction needs reliable, complete output, not
            # creative reasoning, so thinking is disabled entirely.
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
                # 429 (ClientError, quota/rate limit) and 503 (ServerError,
                # transient "high demand") are both retryable -- caught the
                # hard way during the first live run, where an uncaught
                # ServerError (a different exception class than
                # ClientError) crashed the whole multi-hundred-call run on
                # the very first transient 503. Anything else (e.g. a 400
                # bad request) is not transient and must raise immediately.
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
            raise ValueError("Gemini structured completion returned no content")

        import json

        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as exc:
            finish_reason = getattr(getattr(response, "candidates", [None])[0], "finish_reason", None)
            raise ValueError(
                f"Gemini structured completion was not valid JSON (finish_reason={finish_reason!r}, "
                f"likely truncated by max_output_tokens={self._config.max_tokens} -- raise it): {exc}"
            ) from exc

        return StructuredCompletion(
            data=data,
            model_fingerprint=self._config.fingerprint(),
            response_id=getattr(response, "response_id", None) or f"gemini-{int(time.time() * 1000)}",
        )
