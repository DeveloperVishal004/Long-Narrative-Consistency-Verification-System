"""Claim decomposition: the ClaimDecomposer protocol and its LLM-backed implementation."""

from typing import Protocol, runtime_checkable

from lncvs.llm import LLMClient
from lncvs.reasoning.decomposition.config import DecompositionConfig
from lncvs.reasoning.decomposition.identity import make_source_claim_id
from lncvs.reasoning.decomposition.parser import parse_decomposition_response
from lncvs.reasoning.decomposition.prompts import render_decomposition_prompt
from lncvs.schemas import AtomicClaim


@runtime_checkable
class ClaimDecomposer(Protocol):
    """Contract for converting a narrative claim into atomic claims."""

    def decompose(self, original_claim: str) -> list[AtomicClaim]:
        """Return the atomic claims original_claim decomposes into."""
        ...


class LLMClaimDecomposer:
    """ClaimDecomposer backed by an injected LLMClient.

    Holds no state beyond its injected dependencies. The parser it calls is
    pure, so determinism rests entirely on the LLMClient (real calls are
    non-deterministic unless wrapped in CachingLLMClient; FakeLLMClient is
    deterministic by construction).
    """

    def __init__(self, llm_client: LLMClient, config: DecompositionConfig) -> None:
        self._llm_client = llm_client
        self._config = config

    def decompose(self, original_claim: str) -> list[AtomicClaim]:
        if not original_claim or not original_claim.strip():
            raise ValueError("original_claim must not be empty")

        parent_claim_id = make_source_claim_id(original_claim)
        prompt = render_decomposition_prompt(original_claim)
        completion = self._llm_client.complete(prompt)

        return parse_decomposition_response(
            completion.text, parent_claim_id, self._config.max_atomic_claims
        )
