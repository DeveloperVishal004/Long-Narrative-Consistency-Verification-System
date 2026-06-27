"""Deterministic sliding-window chunking over a RawDocument."""

import hashlib
import logging

from lncvs.chunking.config import ChunkingConfig
from lncvs.schemas import DocumentChunk, RawDocument

logger = logging.getLogger(__name__)


def chunk_document(document: RawDocument, config: ChunkingConfig) -> list[DocumentChunk]:
    """Split a RawDocument's cleaned_text into overlapping DocumentChunks.

    chunk_id is a deterministic content hash of (source_id, char_start,
    char_end, text), so re-chunking identical input always reproduces
    identical IDs. Chapter is always None in Phase 1 — chapter detection is
    not yet implemented.
    """
    text = document.cleaned_text
    step = config.chunk_size - config.overlap

    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(text):
        end = min(start + config.chunk_size, len(text))
        chunk_text = text[start:end]

        chunks.append(
            DocumentChunk(
                chunk_id=_make_chunk_id(document.source_id, start, end, chunk_text),
                text=chunk_text,
                char_start=start,
                char_end=end,
                chapter=None,
                source_id=document.source_id,
            )
        )

        if end == len(text):
            break
        start += step

    logger.info("Chunked document %r into %d chunks", document.source_id, len(chunks))
    return chunks


def _make_chunk_id(source_id: str, char_start: int, char_end: int, text: str) -> str:
    digest_input = f"{source_id}:{char_start}:{char_end}:{text}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()[:16]
