"""Fact Verification: the FactVerifier protocol (Phase H2), its
cross-encoder-backed implementation (Phase H2), its LLM-backed
implementation (Phase H3), and the NLIResult compatibility adapter that
lets the frozen rule engine consume either verifier's output unchanged.
CrossEncoderFactVerifier and LLMFactVerifier are fully interchangeable --
everything downstream depends only on the FactVerifier protocol."""

from lncvs.reasoning.fact_verification.compat import to_nli_results
from lncvs.reasoning.fact_verification.llm_config import FactVerificationConfig
from lncvs.reasoning.fact_verification.llm_verifier import LLMFactVerifier
from lncvs.reasoning.fact_verification.service import CrossEncoderFactVerifier, FactVerifier

__all__ = [
    "CrossEncoderFactVerifier",
    "FactVerificationConfig",
    "FactVerifier",
    "LLMFactVerifier",
    "to_nli_results",
]
