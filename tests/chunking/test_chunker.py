"""Chunker tests: coverage, determinism, overlap, and boundary cases."""

from lncvs.chunking import ChunkingConfig, chunk_document
from lncvs.schemas import RawDocument


def _document(text: str) -> RawDocument:
    return RawDocument(source_id="doc-1", raw_text=text, cleaned_text=text)


def test_short_text_produces_a_single_chunk() -> None:
    document = _document("John lost his left arm in an accident in 2010.")
    config = ChunkingConfig(chunk_size=200, overlap=20)

    chunks = chunk_document(document, config)

    assert len(chunks) == 1
    assert chunks[0].text == document.cleaned_text
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(document.cleaned_text)


def test_chunks_cover_the_full_document_with_overlap() -> None:
    text = "abcdefghij" * 10  # 100 chars
    document = _document(text)
    config = ChunkingConfig(chunk_size=30, overlap=10)

    chunks = chunk_document(document, config)

    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(text)
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.char_start == earlier.char_start + (config.chunk_size - config.overlap)
        assert later.char_start < earlier.char_end  # overlap actually occurs


def test_chunk_ids_are_deterministic_across_runs() -> None:
    document = _document("John lost his left arm in an accident in 2010. John moved to London in 2012.")
    config = ChunkingConfig(chunk_size=40, overlap=10)

    first_run = chunk_document(document, config)
    second_run = chunk_document(document, config)

    assert [c.chunk_id for c in first_run] == [c.chunk_id for c in second_run]


def test_no_chunk_exceeds_configured_chunk_size() -> None:
    document = _document("John lost his left arm in an accident in 2010. John moved to London in 2012.")
    config = ChunkingConfig(chunk_size=40, overlap=10)

    chunks = chunk_document(document, config)

    assert all(len(chunk.text) <= config.chunk_size for chunk in chunks)


def test_chunking_preserves_the_target_sentence() -> None:
    """A chunk containing the dummy-case sentence must exist somewhere in the output."""
    document = _document("John lost his left arm in an accident in 2010.\n\nJohn moved to London in 2012.")
    config = ChunkingConfig(chunk_size=200, overlap=20)

    chunks = chunk_document(document, config)

    assert any("John lost his left arm in an accident" in chunk.text for chunk in chunks)
