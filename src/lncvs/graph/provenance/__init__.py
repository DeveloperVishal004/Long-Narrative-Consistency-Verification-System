"""Provenance assignment (Phase 8 / G2 Slice 4): the trust boundary between
LLM extraction output and the deterministic graph.

Not re-exported from lncvs.graph's top-level __init__ -- callers import
from lncvs.graph.provenance directly, the same convention
lncvs.graph.llm_extraction and lncvs.reasoning.decomposition follow.
"""

from lncvs.graph.provenance.canon import canonicalize_with_offsets
from lncvs.graph.provenance.config import ProvenanceConfig
from lncvs.graph.provenance.matching import MatchTier, QuoteMatch, resolve_quote
from lncvs.graph.provenance.models import RawFact, RejectedFact, ResolvedFact, WindowProvenanceResult
from lncvs.graph.provenance.service import resolve_window_provenance

__all__ = [
    "MatchTier",
    "ProvenanceConfig",
    "QuoteMatch",
    "RawFact",
    "RejectedFact",
    "ResolvedFact",
    "WindowProvenanceResult",
    "canonicalize_with_offsets",
    "resolve_quote",
    "resolve_window_provenance",
]
