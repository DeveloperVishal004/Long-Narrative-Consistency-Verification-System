"""NLI Verification: verifies atomic claims against fused evidence, evidence-level only."""

from lncvs.reasoning.nli.cache import CachingNLIModel, InMemoryNLICache, NLICache
from lncvs.reasoning.nli.config import NLIConfig
from lncvs.reasoning.nli.model import CrossEncoderNLIModel, NLIModel, NLIPrediction
from lncvs.reasoning.nli.service import CrossEncoderNLIVerifier

__all__ = [
    "CachingNLIModel",
    "CrossEncoderNLIModel",
    "CrossEncoderNLIVerifier",
    "InMemoryNLICache",
    "NLICache",
    "NLIConfig",
    "NLIModel",
    "NLIPrediction",
]
