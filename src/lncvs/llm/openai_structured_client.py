"""OpenAIStructuredClient -- the only module in this codebase that imports
the openai SDK, mirroring the isolation discipline already applied to
chromadb (indexing/chroma_index.py), rank_bm25 (indexing/bm25_index.py),
and networkx (graph/builder.py).

Implements StructuredLLMClient via the Chat Completions API's
response_format={"type": "json_schema", ..., "strict": True} mode -- the
frozen G2 extraction interface's "provider-enforced structured outputs"
requirement (§7 of the interface freeze). temperature/max_tokens come from
the injected LLMConfig; the system prompt is fixed at construction time
(it is provider-client configuration, not a per-call parameter), which is
what lets this class satisfy StructuredLLMClient's single-prompt
complete_structured(prompt, response_schema) signature unchanged while
still using a real system role at the HTTP layer.
"""

import json
import logging

from lncvs.llm.config import LLMConfig
from lncvs.llm.structured import StructuredCompletion

logger = logging.getLogger(__name__)


class OpenAIStructuredClient:
    """StructuredLLMClient backed by the real OpenAI API.

    api_key defaults to None, which lets the underlying openai.OpenAI()
    client fall back to the OPENAI_API_KEY environment variable -- no
    secret material is ever read or logged by this class itself.
    """

    def __init__(self, config: LLMConfig, system_prompt: str, schema_name: str, api_key: str | None = None) -> None:
        import openai

        self._client = openai.OpenAI(api_key=api_key)
        self._config = config
        self._system_prompt = system_prompt
        self._schema_name = schema_name

    def complete_structured(self, prompt: str, response_schema: dict) -> StructuredCompletion:
        response = self._client.chat.completions.create(
            model=self._config.model_name,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": self._schema_name, "strict": True, "schema": response_schema},
            },
        )

        choice = response.choices[0]
        if choice.finish_reason != "stop":
            logger.error("OpenAI structured completion did not finish cleanly: %r", choice.finish_reason)
            raise ValueError(f"OpenAI structured completion did not finish cleanly: finish_reason={choice.finish_reason!r}")

        raw_text = choice.message.content
        if raw_text is None:
            raise ValueError("OpenAI structured completion returned no content")

        data = json.loads(raw_text)  # guaranteed valid JSON conforming to response_schema under strict mode

        return StructuredCompletion(
            data=data,
            model_fingerprint=self._config.fingerprint(),
            response_id=response.id,
        )
