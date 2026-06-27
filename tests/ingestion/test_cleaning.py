"""Text cleaning tests."""

from lncvs.ingestion import clean_text


def test_clean_text_normalizes_crlf_line_endings() -> None:
    assert clean_text("line one\r\nline two\r\n") == "line one\nline two"


def test_clean_text_strips_bom() -> None:
    assert clean_text("﻿John lost his left arm.") == "John lost his left arm."


def test_clean_text_strips_trailing_whitespace_per_line() -> None:
    assert clean_text("line one   \nline two\t\n") == "line one\nline two"


def test_clean_text_collapses_excess_blank_lines() -> None:
    assert clean_text("para one\n\n\n\n\npara two") == "para one\n\npara two"


def test_clean_text_strips_leading_and_trailing_whitespace() -> None:
    assert clean_text("\n\n  John lost his left arm.  \n\n") == "John lost his left arm."
