"""Deterministic, offline test doubles for the LLMClient and
StructuredLLMClient protocols."""

from lncvs.llm import LLMCompletion, StructuredCompletion


class FakeLLMClient:
    """A scripted LLMClient: returns a fixed response for known prompts, or a
    default response for any prompt if one is configured. Records every
    prompt it was called with, so tests can assert call counts (e.g. to
    prove a CachingLLMClient avoided a redundant call)."""

    def __init__(
        self,
        scripted: dict[str, str] | None = None,
        default_response: str | None = None,
        model_fingerprint: str = "fake-model",
    ) -> None:
        self._scripted = scripted or {}
        self._default_response = default_response
        self._model_fingerprint = model_fingerprint
        self.calls: list[str] = []

    def complete(self, prompt: str) -> LLMCompletion:
        self.calls.append(prompt)

        if prompt in self._scripted:
            text = self._scripted[prompt]
        elif self._default_response is not None:
            text = self._default_response
        else:
            raise ValueError(f"FakeLLMClient has no scripted response for prompt: {prompt!r}")

        return LLMCompletion(text=text, model_fingerprint=self._model_fingerprint)


class FakeStructuredLLMClient:
    """A scripted StructuredLLMClient: returns a fixed dict response for
    known prompts, or a default response for any prompt if one is
    configured. Records every (prompt, response_schema) call it received,
    so tests can assert call counts and that the right schema was passed."""

    def __init__(
        self,
        scripted: dict[str, dict] | None = None,
        default_response: dict | None = None,
        model_fingerprint: str = "fake-structured-model",
    ) -> None:
        self._scripted = scripted or {}
        self._default_response = default_response
        self._model_fingerprint = model_fingerprint
        self.calls: list[tuple[str, dict]] = []

    def complete_structured(self, prompt: str, response_schema: dict) -> StructuredCompletion:
        self.calls.append((prompt, response_schema))

        if prompt in self._scripted:
            data = self._scripted[prompt]
        elif self._default_response is not None:
            data = self._default_response
        else:
            raise ValueError(f"FakeStructuredLLMClient has no scripted response for prompt: {prompt!r}")

        return StructuredCompletion(
            data=data,
            model_fingerprint=self._model_fingerprint,
            response_id=f"fake-response-{len(self.calls)}",
        )
