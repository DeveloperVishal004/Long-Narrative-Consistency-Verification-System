"""Closed-vocabulary tests: verdicts, NLI labels, retrieval sources, pipeline stages."""

from lncvs.schemas import NLILabel, PipelineStage, RetrievalSource, VerdictEnum


def test_verdict_enum_has_exactly_three_members() -> None:
    assert {member.value for member in VerdictEnum} == {
        "CONSISTENT",
        "CONTRADICTORY",
        "INSUFFICIENT_EVIDENCE",
    }


def test_nli_label_has_exactly_three_members() -> None:
    assert {member.value for member in NLILabel} == {"ENTAILMENT", "CONTRADICTION", "NEUTRAL"}


def test_retrieval_source_has_exactly_three_members_including_graph() -> None:
    """GRAPH was added in Phase 8 (Version 2): an explicit, recorded
    project-owner decision that the V2 entry gate is satisfied and graph
    work is authorized in this repository -- see schemas/enums.py's
    RetrievalSource docstring. It is produced only by the opt-in
    lncvs.graph.GraphRetriever; no V1 code path wires it in by default."""
    values = {member.value for member in RetrievalSource}
    assert values == {"SEMANTIC", "LEXICAL", "GRAPH"}


def test_pipeline_stage_covers_the_v1_pipeline() -> None:
    values = {member.value for member in PipelineStage}
    expected = {
        "INGESTION",
        "CHUNKING",
        "INDEXING",
        "CLAIM_DECOMPOSITION",
        "QUESTION_GENERATION",
        "RETRIEVAL",
        "FUSION",
        "NLI_VERIFICATION",
        "RULE_ENGINE",
        "COMPLETE",
        "ERROR",
    }
    assert values == expected
