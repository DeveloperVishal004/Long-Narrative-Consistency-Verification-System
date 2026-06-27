"""Deterministic cross-window entity resolution (Phase 8 / G2 Slice 5).

Not re-exported from lncvs.graph's top-level __init__ -- callers import
from lncvs.graph.entity_resolution directly, the same convention
lncvs.graph.llm_extraction and lncvs.graph.provenance follow.
"""

from lncvs.graph.entity_resolution.merge import (
    compute_components,
    merge_component,
    select_canonical_name,
    select_entity_type,
)
from lncvs.graph.entity_resolution.models import EntityResolutionResult, LocalEntityKey
from lncvs.graph.entity_resolution.normalization import is_generic_referent, norm_name
from lncvs.graph.entity_resolution.service import resolve_entities

__all__ = [
    "EntityResolutionResult",
    "LocalEntityKey",
    "compute_components",
    "is_generic_referent",
    "merge_component",
    "norm_name",
    "resolve_entities",
    "select_canonical_name",
    "select_entity_type",
]
