"""Fact-verification prompt rendering tests (evidence-SET-level redesign)."""

import inspect

import pytest

from lncvs.reasoning.fact_verification.llm_prompts import PROMPT_VERSION, SYSTEM_PROMPT, render_fact_verification_prompt


def test_rendered_prompt_contains_fact_and_all_evidence_passages() -> None:
    rendered = render_fact_verification_prompt(
        "John lost his left arm.", ["John lost his left arm in 2010.", "John moved to London in 2012."]
    )

    assert "John lost his left arm." in rendered
    assert "John lost his left arm in 2010." in rendered
    assert "John moved to London in 2012." in rendered


def test_rendered_prompt_labels_each_passage() -> None:
    rendered = render_fact_verification_prompt("fact", ["passage one", "passage two", "passage three"])

    assert "PASSAGE 1" in rendered
    assert "PASSAGE 2" in rendered
    assert "PASSAGE 3" in rendered


def test_rendered_prompt_states_the_passage_count() -> None:
    rendered = render_fact_verification_prompt("fact", ["a", "b", "c"])

    assert "3 evidence passage" in rendered


def test_render_rejects_empty_evidence_list() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        render_fact_verification_prompt("fact", [])


def test_render_takes_only_fact_and_evidence_no_backstory_parameter() -> None:
    """Structural guarantee, not just a convention: render_fact_verification_prompt
    has exactly two parameters. There is no third argument through which an
    original backstory could be smuggled in."""
    params = list(inspect.signature(render_fact_verification_prompt).parameters)
    assert params == ["fact_text", "evidence_texts"]


def test_system_prompt_states_not_mentioned_is_not_contradicted() -> None:
    assert "NOT_MENTIONED" in SYSTEM_PROMPT
    assert "NEVER" in SYSTEM_PROMPT or "never" in SYSTEM_PROMPT.lower()
    assert "contradiction" in SYSTEM_PROMPT.lower() or "CONTRADICTED" in SYSTEM_PROMPT


def test_system_prompt_requires_verbatim_quotes_for_supported_and_contradicted() -> None:
    assert "verbatim" in SYSTEM_PROMPT.lower()
    assert "SUPPORTED" in SYSTEM_PROMPT
    assert "CONTRADICTED" in SYSTEM_PROMPT


def test_system_prompt_forbids_outside_knowledge() -> None:
    assert "outside knowledge" in SYSTEM_PROMPT.lower()


def test_system_prompt_instructs_reasoning_across_the_complete_set() -> None:
    """The core instruction motivating the redesign: judge the complete
    evidence set together, not any single passage in isolation."""
    assert "complete" in SYSTEM_PROMPT.lower() or "together" in SYSTEM_PROMPT.lower()
    assert "set" in SYSTEM_PROMPT.lower()


def test_prompt_version_is_stable_hash() -> None:
    assert len(PROMPT_VERSION) == 8
    assert PROMPT_VERSION == PROMPT_VERSION
