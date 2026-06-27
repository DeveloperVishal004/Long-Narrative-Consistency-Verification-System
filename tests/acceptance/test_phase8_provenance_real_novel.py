"""Phase 8 / G2 Slice 4 acceptance test: provenance assignment against the
real Castaways narrative -- real chunks (the project's established
chunk_size=700/overlap=120), a real chapter window, and verbatim quotes
taken directly from the real text (no LLM call -- cheap, always run, not
gated).

Proves the tiered matching + chunk-overlap resolution work end-to-end on
genuine prose, not just hand-built fixtures: a quote near a real chunk
boundary correctly resolves to multiple real chunk_ids, and a fabricated
quote is correctly quarantined.
"""

from pathlib import Path

import pytest

from lncvs.chunking import ChunkingConfig, chunk_document
from lncvs.graph.llm_extraction.schema import RawEntityMention, WindowExtraction
from lncvs.graph.provenance.service import resolve_window_provenance
from lncvs.graph.segmentation import segment_into_chapters
from lncvs.ingestion import load_and_clean_narrative
from lncvs.schemas import EntityType

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
NARRATIVE_PATH = DATA_DIR / "In search of the castaways.txt"


@pytest.fixture(scope="module")
def real_chapter_fixture():
    if not NARRATIVE_PATH.exists():
        pytest.skip("Castaways narrative not present in data/")

    document = load_and_clean_narrative(NARRATIVE_PATH, source_id="castaways")
    chapters = segment_into_chapters(document.cleaned_text)
    chunks = chunk_document(document, ChunkingConfig(chunk_size=700, overlap=120))

    first_chapter = chapters[0]
    chapter_text = document.cleaned_text[first_chapter.char_start : first_chapter.char_end]
    return chapter_text, first_chapter.char_start, chunks


def test_real_verbatim_quote_resolves_to_real_chunks(real_chapter_fixture) -> None:
    chapter_text, char_start, chunks = real_chapter_fixture

    # Take a genuine, real substring directly from the chapter as the quote.
    midpoint = len(chapter_text) // 2
    real_quote = chapter_text[midpoint : midpoint + 60].strip()
    if len(real_quote) < 20:
        real_quote = chapter_text[midpoint : midpoint + 120].strip()

    extraction = WindowExtraction(
        entities=(RawEntityMention(local_id="e1", name="RealEntity", type=EntityType.OTHER, evidence_quotes=(real_quote,)),)
    )
    result = resolve_window_provenance(extraction, chapter_text, 0, None, char_start, chunks)

    assert len(result.resolved_entities) == 1
    resolved = result.resolved_entities[0]
    assert len(resolved.provenance) >= 1
    for provenance in resolved.provenance:
        chunk = next(c for c in chunks if c.chunk_id == provenance.chunk_id)
        assert chunk.char_start <= provenance.char_start < provenance.char_end <= chunk.char_end


def test_fabricated_quote_against_real_chapter_is_quarantined(real_chapter_fixture) -> None:
    chapter_text, char_start, chunks = real_chapter_fixture

    extraction = WindowExtraction(
        entities=(
            RawEntityMention(
                local_id="e1",
                name="Imaginary",
                type=EntityType.OTHER,
                evidence_quotes=("a sentence that absolutely does not appear anywhere in this novel chapter",),
            ),
        )
    )
    result = resolve_window_provenance(extraction, chapter_text, 0, None, char_start, chunks)

    assert len(result.resolved_entities) == 0
    assert len(result.rejected_entities) == 1


def test_quote_at_a_real_chunk_boundary_resolves_to_multiple_chunks(real_chapter_fixture) -> None:
    chapter_text, char_start, chunks = real_chapter_fixture
    sorted_chunks = sorted(chunks, key=lambda c: c.char_start)
    if len(sorted_chunks) < 2:
        pytest.skip("First chapter too short to span multiple chunks at chunk_size=700")

    # A real boundary between the first two chunks, given overlap=120.
    boundary = sorted_chunks[1].char_start
    local_boundary = boundary - char_start
    quote = chapter_text[max(0, local_boundary - 30) : local_boundary + 30].strip()

    extraction = WindowExtraction(
        entities=(RawEntityMention(local_id="e1", name="BoundaryEntity", type=EntityType.OTHER, evidence_quotes=(quote,)),)
    )
    result = resolve_window_provenance(extraction, chapter_text, 0, None, char_start, chunks)

    assert len(result.resolved_entities) == 1
    chunk_ids = {p.chunk_id for p in result.resolved_entities[0].provenance}
    assert len(chunk_ids) >= 2
