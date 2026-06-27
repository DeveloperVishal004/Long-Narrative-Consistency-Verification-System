"""Hackathon-mode fix: a single window's extraction failure (e.g. a
deterministic MAX_TOKENS truncation on one unusually dense chapter, seen
during a live Monte Cristo run) must not abort graph construction for the
entire novel. Mirrors the existing "one backend failing must not crash the
run" discipline already applied to retrieval -- the failed window simply
contributes zero entities/relations/events, logged loudly, not silently."""

from lncvs.graph.construction.pipeline import build_graph_for_novel
from lncvs.graph.llm_extraction.schema import WindowExtraction
from lncvs.schemas import DocumentChunk

NARRATIVE = (
    "Chapter 1\nJohn lost his left arm in an accident in 2010.\n\n"
    "Chapter 2\nMary moved to Paris in 2011.\n\n"
    "Chapter 3\nMary married John in 2012.\n"
)

CHUNKS = [DocumentChunk(chunk_id="chunk-0", text=NARRATIVE, char_start=0, char_end=len(NARRATIVE), source_id="dummy")]


class _FailOnChapterExtractor:
    """A minimal WindowExtractor that raises ValueError for one specific
    chapter index and returns a real entity for every other chapter --
    isolates the fault-isolation behavior without needing real provider
    plumbing."""

    def __init__(self, fail_chapter_index: int) -> None:
        self._fail_chapter_index = fail_chapter_index
        self.calls: list[int] = []

    def extract(self, window_text: str, chapter_index: int, window_index: int | None) -> WindowExtraction:
        self.calls.append(chapter_index)
        if chapter_index == self._fail_chapter_index:
            raise ValueError("Gemini structured completion was not valid JSON (simulated MAX_TOKENS truncation)")
        return WindowExtraction.model_validate(
            {
                "entities": [
                    {
                        "local_id": "e1",
                        "name": f"Person{chapter_index}",
                        "type": "PERSON",
                        "aliases": [],
                        "evidence_quotes": [window_text.strip().splitlines()[-1]],
                    }
                ],
                "relations": [],
            }
        )


def test_one_failed_window_is_skipped_not_fatal() -> None:
    extractor = _FailOnChapterExtractor(fail_chapter_index=1)

    constructed, entity_graph = build_graph_for_novel(NARRATIVE, CHUNKS, extractor)

    # All three chapters were attempted (the failure did not stop iteration).
    assert extractor.calls == [0, 1, 2]
    # Chapter 1's entity never appears -- it contributed zero facts.
    names = {e.canonical_name for e in constructed.entities}
    assert "Person1" not in names
    # Chapters 0 and 2 still succeeded and reached the graph.
    assert "Person0" in names
    assert "Person2" in names
    assert entity_graph.entity_count() == 2


def test_no_failures_means_no_windows_skipped() -> None:
    extractor = _FailOnChapterExtractor(fail_chapter_index=999)

    constructed, _ = build_graph_for_novel(NARRATIVE, CHUNKS, extractor)

    assert extractor.calls == [0, 1, 2]
    assert len(constructed.entities) == 3
