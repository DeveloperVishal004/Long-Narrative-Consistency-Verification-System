"""LLM-backed structured extraction over a single chapter/window (Phase 8 / G2).

Not re-exported from lncvs.graph's top-level __init__ -- callers import
from lncvs.graph.llm_extraction directly, the same convention
lncvs.reasoning.decomposition follows relative to lncvs.reasoning.
"""

from lncvs.graph.llm_extraction.config import ExtractionConfig
from lncvs.graph.llm_extraction.json_schema import EXTRACTION_JSON_SCHEMA, SCHEMA_VERSION
from lncvs.graph.llm_extraction.prompts import PROMPT_VERSION, SYSTEM_PROMPT, render_user_prompt
from lncvs.graph.llm_extraction.schema import (
    RawEntityMention,
    RawEvent,
    RawParticipant,
    RawRelation,
    RawTemporal,
    WindowExtraction,
)
from lncvs.graph.llm_extraction.service import LLMWindowExtractor, WindowExtractor, parse_window_extraction

__all__ = [
    "EXTRACTION_JSON_SCHEMA",
    "ExtractionConfig",
    "LLMWindowExtractor",
    "PROMPT_VERSION",
    "RawEntityMention",
    "RawEvent",
    "RawParticipant",
    "RawRelation",
    "RawTemporal",
    "SCHEMA_VERSION",
    "SYSTEM_PROMPT",
    "WindowExtraction",
    "WindowExtractor",
    "parse_window_extraction",
    "render_user_prompt",
]
