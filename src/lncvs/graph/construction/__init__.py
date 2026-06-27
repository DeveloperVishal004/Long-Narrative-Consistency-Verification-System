"""Graph construction (Phase 8 / G2 Slice 6): assembles resolved relations
and events into the final typed graph content, with dangling-reference
quarantine and a deterministic graph fingerprint.

Not re-exported from lncvs.graph's top-level __init__ -- callers import
from lncvs.graph.construction directly, the same convention every other
G2 submodule follows.
"""

from lncvs.graph.construction.fingerprint import compute_graph_fingerprint
from lncvs.graph.construction.models import ConstructedGraph
from lncvs.graph.construction.pipeline import build_graph_for_novel
from lncvs.graph.construction.service import build_graph

__all__ = ["ConstructedGraph", "build_graph", "build_graph_for_novel", "compute_graph_fingerprint"]
