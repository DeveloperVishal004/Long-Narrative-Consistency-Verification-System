"""Claim Decomposition: converts a narrative claim into deterministic, traceable atomic claims."""

from lncvs.reasoning.decomposition.config import DecompositionConfig
from lncvs.reasoning.decomposition.identity import make_atomic_claim_id, make_source_claim_id
from lncvs.reasoning.decomposition.parser import parse_decomposition_response
from lncvs.reasoning.decomposition.prompts import PROMPT_VERSION, render_decomposition_prompt
from lncvs.reasoning.decomposition.service import ClaimDecomposer, LLMClaimDecomposer

__all__ = [
    "ClaimDecomposer",
    "DecompositionConfig",
    "LLMClaimDecomposer",
    "PROMPT_VERSION",
    "make_atomic_claim_id",
    "make_source_claim_id",
    "parse_decomposition_response",
    "render_decomposition_prompt",
]
