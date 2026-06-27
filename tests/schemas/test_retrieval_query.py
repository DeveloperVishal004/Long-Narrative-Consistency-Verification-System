"""RetrievalQuery validation tests."""

import pytest
from pydantic import ValidationError

from lncvs.schemas import QueryOrigin, RetrievalQuery


def test_claim_origin_query_valid_construction() -> None:
    query = RetrievalQuery(
        query_id="query-1", text="John used both hands", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM
    )
    assert query.question_id is None


def test_question_origin_query_valid_construction() -> None:
    query = RetrievalQuery(
        query_id="query-1",
        text="Did John lose an arm?",
        atomic_claim_id="claim-1",
        question_id="q-1",
        origin=QueryOrigin.QUESTION,
    )
    assert query.question_id == "q-1"


def test_claim_origin_with_question_id_set_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must be None when origin is CLAIM"):
        RetrievalQuery(
            query_id="query-1",
            text="John used both hands",
            atomic_claim_id="claim-1",
            question_id="q-1",
            origin=QueryOrigin.CLAIM,
        )


def test_question_origin_without_question_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must be set when origin is QUESTION"):
        RetrievalQuery(
            query_id="query-1",
            text="Did John lose an arm?",
            atomic_claim_id="claim-1",
            origin=QueryOrigin.QUESTION,
        )


def test_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(query_id="query-1", text="", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM)
