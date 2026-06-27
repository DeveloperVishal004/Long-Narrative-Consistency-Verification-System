"""Chunking: deterministic, content-hash-identified sliding-window chunking."""

from lncvs.chunking.chunker import chunk_document
from lncvs.chunking.config import ChunkingConfig
from lncvs.chunking.span_overlap import chunks_overlapping_any_span, chunks_overlapping_span

__all__ = ["ChunkingConfig", "chunk_document", "chunks_overlapping_any_span", "chunks_overlapping_span"]
