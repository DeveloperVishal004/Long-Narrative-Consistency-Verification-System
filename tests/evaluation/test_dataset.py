"""load_dataset() and map_spans_to_chunks() tests."""

from pathlib import Path

import pytest

from lncvs.evaluation import load_dataset, map_spans_to_chunks
from lncvs.schemas import DocumentChunk, GoldSpan


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_dataset_parses_one_example_per_line(tmp_path: Path) -> None:
    dataset_path = tmp_path / "gold.jsonl"
    _write_jsonl(
        dataset_path,
        [
            '{"example_id": "ex-1", "narrative_path": "a.txt", "original_claim": "claim a", "expected_verdict": "CONSISTENT"}',
            '{"example_id": "ex-2", "narrative_path": "b.txt", "original_claim": "claim b", "expected_verdict": "CONTRADICTORY"}',
        ],
    )

    dataset = load_dataset(dataset_path, dataset_id="test-ds")

    assert dataset.dataset_id == "test-ds"
    assert len(dataset.examples) == 2
    assert dataset.examples[0].example_id == "ex-1"


def test_load_dataset_skips_blank_lines(tmp_path: Path) -> None:
    dataset_path = tmp_path / "gold.jsonl"
    _write_jsonl(
        dataset_path,
        [
            '{"example_id": "ex-1", "narrative_path": "a.txt", "original_claim": "claim a", "expected_verdict": "CONSISTENT"}',
            "",
            "   ",
        ],
    )

    dataset = load_dataset(dataset_path, dataset_id="test-ds")
    assert len(dataset.examples) == 1


def test_load_dataset_raises_on_empty_file(tmp_path: Path) -> None:
    dataset_path = tmp_path / "empty.jsonl"
    dataset_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no examples"):
        load_dataset(dataset_path, dataset_id="test-ds")


def _chunk(chunk_id: str, start: int, end: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id, text="x" * (end - start), char_start=start, char_end=end, source_id="narr-1"
    )


def test_map_spans_to_chunks_finds_overlapping_chunks() -> None:
    chunks = [_chunk("chunk-0", 0, 50), _chunk("chunk-1", 40, 90), _chunk("chunk-2", 100, 150)]
    spans = [GoldSpan(char_start=45, char_end=48)]

    relevant = map_spans_to_chunks(spans, chunks)

    assert relevant == {"chunk-0", "chunk-1"}


def test_map_spans_to_chunks_returns_empty_set_for_no_overlap() -> None:
    chunks = [_chunk("chunk-0", 0, 50)]
    spans = [GoldSpan(char_start=100, char_end=110)]

    relevant = map_spans_to_chunks(spans, chunks)

    assert relevant == set()


def test_map_spans_to_chunks_handles_multiple_spans() -> None:
    chunks = [_chunk("chunk-0", 0, 50), _chunk("chunk-1", 50, 100)]
    spans = [GoldSpan(char_start=10, char_end=20), GoldSpan(char_start=60, char_end=70)]

    relevant = map_spans_to_chunks(spans, chunks)

    assert relevant == {"chunk-0", "chunk-1"}
