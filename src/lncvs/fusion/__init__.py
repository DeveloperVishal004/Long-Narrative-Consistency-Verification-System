"""Fusion: pure Reciprocal Rank Fusion over claim-linked retrieval evidence."""

from lncvs.fusion.config import FusionConfig
from lncvs.fusion.rrf import fuse_evidence

__all__ = ["FusionConfig", "fuse_evidence"]
