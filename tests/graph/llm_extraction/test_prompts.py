"""Extraction prompt rendering and versioning."""

from lncvs.graph.llm_extraction.prompts import PROMPT_VERSION, SYSTEM_PROMPT, render_user_prompt


def test_prompt_version_is_deterministic() -> None:
    assert PROMPT_VERSION == PROMPT_VERSION
    assert len(PROMPT_VERSION) == 8


def test_system_prompt_states_the_hallucination_prevention_rules() -> None:
    assert "VERBATIM" in SYSTEM_PROMPT
    assert "DO NOT output that fact" in SYSTEM_PROMPT
    assert "outside the provided passage" in SYSTEM_PROMPT


def test_render_user_prompt_without_window_index() -> None:
    rendered = render_user_prompt("Some passage text.", chapter_index=3, window_index=None)
    assert "chapter 3 of a novel" in rendered
    assert "window" not in rendered.split("PASSAGE")[0]
    assert "Some passage text." in rendered


def test_render_user_prompt_with_window_index() -> None:
    rendered = render_user_prompt("Some passage text.", chapter_index=3, window_index=2)
    assert "chapter 3, window 2 of a novel" in rendered


def test_render_user_prompt_is_deterministic() -> None:
    a = render_user_prompt("Text.", chapter_index=1, window_index=0)
    b = render_user_prompt("Text.", chapter_index=1, window_index=0)
    assert a == b
