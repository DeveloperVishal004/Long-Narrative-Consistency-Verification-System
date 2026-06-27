"""LLM client infrastructure: provider-agnostic protocol, config, and caching.

LLMClient/complete() is shared by Claim Decomposition (Phase 2a, activated
for real in Phase H1) and Question Generation (Phase 2b). GeminiLLMClient
is its first concrete provider implementation; FakeLLMClient test doubles
remain the offline-test path. StructuredLLMClient/complete_structured() is
a separate, additive protocol introduced in Phase 8 (G2) solely for
schema-enforced structured extraction -- see structured.py's docstring for
why it does not replace or modify LLMClient. NLI does not use either
protocol; it has its own dedicated abstraction.
"""

from lncvs.llm.base import LLMClient, LLMCompletion
from lncvs.llm.cache import CachingLLMClient, InMemoryLLMCache, JsonlLLMCache, LLMCache
from lncvs.llm.config import LLMConfig
from lncvs.llm.gemini_llm_client import GeminiLLMClient
from lncvs.llm.gemini_structured_client import GeminiStructuredClient
from lncvs.llm.openai_structured_client import OpenAIStructuredClient
from lncvs.llm.structured import StructuredCompletion, StructuredLLMClient
from lncvs.llm.structured_cache import CachingStructuredLLMClient, InMemoryStructuredLLMCache, JsonlStructuredLLMCache, StructuredLLMCache

__all__ = [
    "CachingLLMClient",
    "CachingStructuredLLMClient",
    "GeminiLLMClient",
    "GeminiStructuredClient",
    "InMemoryLLMCache",
    "InMemoryStructuredLLMCache",
    "JsonlLLMCache",
    "JsonlStructuredLLMCache",
    "LLMCache",
    "LLMClient",
    "LLMCompletion",
    "LLMConfig",
    "OpenAIStructuredClient",
    "StructuredCompletion",
    "StructuredLLMCache",
    "StructuredLLMClient",
]
