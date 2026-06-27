"""build_retrieval_queries tests — pure, no retriever involved."""

from lncvs.retrieval import build_retrieval_queries
from lncvs.schemas import AtomicClaim, ProbeQuestion, QueryOrigin

CLAIM_A = AtomicClaim(claim_id="claim-a", text="John played piano", parent_claim_id="parent-1", index=0)
CLAIM_B = AtomicClaim(claim_id="claim-b", text="John used both hands", parent_claim_id="parent-1", index=1)


def test_one_claim_query_per_atomic_claim() -> None:
    queries = build_retrieval_queries([CLAIM_A, CLAIM_B], [])

    assert len(queries) == 2
    assert all(q.origin is QueryOrigin.CLAIM for q in queries)
    assert all(q.question_id is None for q in queries)
    assert {q.atomic_claim_id for q in queries} == {"claim-a", "claim-b"}
    assert {q.text for q in queries} == {"John played piano", "John used both hands"}


def test_one_question_query_per_probe_question() -> None:
    question = ProbeQuestion(question_id="q-1", atomic_claim_id="claim-b", text="Did John lose an arm?", index=0)

    queries = build_retrieval_queries([CLAIM_B], [question])

    question_queries = [q for q in queries if q.origin is QueryOrigin.QUESTION]
    assert len(question_queries) == 1
    assert question_queries[0].question_id == "q-1"
    assert question_queries[0].atomic_claim_id == "claim-b"
    assert question_queries[0].text == "Did John lose an arm?"


def test_claim_with_no_probe_questions_still_gets_a_claim_query() -> None:
    """The fallback query — a claim with zero probe questions is non-fatal."""
    queries = build_retrieval_queries([CLAIM_A], [])

    assert len(queries) == 1
    assert queries[0].origin is QueryOrigin.CLAIM


def test_query_ids_are_deterministic_across_calls() -> None:
    question = ProbeQuestion(question_id="q-1", atomic_claim_id="claim-b", text="Did John lose an arm?")

    first_run = build_retrieval_queries([CLAIM_A, CLAIM_B], [question])
    second_run = build_retrieval_queries([CLAIM_A, CLAIM_B], [question])

    assert [q.query_id for q in first_run] == [q.query_id for q in second_run]


def test_claim_and_question_query_ids_never_collide_even_with_identical_text() -> None:
    """A probe question whose text happens to equal its own claim's text must
    still produce a distinct query_id (origin is folded into the hash)."""
    claim = AtomicClaim(claim_id="claim-c", text="Did John lose an arm?", parent_claim_id="parent-1")
    question = ProbeQuestion(question_id="q-1", atomic_claim_id="claim-c", text="Did John lose an arm?")

    queries = build_retrieval_queries([claim], [question])

    assert len(queries) == 2
    assert queries[0].query_id != queries[1].query_id


def test_duplicate_probe_questions_are_deduplicated_by_query_id() -> None:
    question = ProbeQuestion(question_id="q-1", atomic_claim_id="claim-b", text="Did John lose an arm?")

    queries = build_retrieval_queries([CLAIM_B], [question, question])

    question_queries = [q for q in queries if q.origin is QueryOrigin.QUESTION]
    assert len(question_queries) == 1


def test_multiple_claims_and_questions_preserve_input_order_claims_first() -> None:
    question_a = ProbeQuestion(question_id="q-a", atomic_claim_id="claim-a", text="Did John play an instrument?")
    question_b = ProbeQuestion(question_id="q-b", atomic_claim_id="claim-b", text="Did John lose an arm?")

    queries = build_retrieval_queries([CLAIM_A, CLAIM_B], [question_a, question_b])

    origins = [q.origin for q in queries]
    assert origins == [QueryOrigin.CLAIM, QueryOrigin.CLAIM, QueryOrigin.QUESTION, QueryOrigin.QUESTION]
