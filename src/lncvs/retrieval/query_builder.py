"""Pure builder: atomic claims + probe questions -> unified RetrievalQuery list.

No retriever calls happen here. One CLAIM-origin query per atomic claim
(always present — the fallback query when a claim has no probe questions)
plus one QUESTION-origin query per probe question.
"""

from lncvs.retrieval.identity import make_query_id
from lncvs.schemas import AtomicClaim, ProbeQuestion, QueryOrigin, RetrievalQuery


def build_retrieval_queries(
    atomic_claims: list[AtomicClaim], probe_questions: list[ProbeQuestion]
) -> list[RetrievalQuery]:
    """Build the full set of retrieval queries for a decomposed claim.

    Deduplicates by query_id (defensive — the identity scheme already makes
    collisions between a CLAIM query and a QUESTION query for the same claim
    impossible, since origin is folded into the hash, but duplicate input
    ProbeQuestions would otherwise produce duplicate queries).
    """
    queries: list[RetrievalQuery] = []
    seen_query_ids: set[str] = set()

    for claim in atomic_claims:
        query_id = make_query_id(claim.claim_id, QueryOrigin.CLAIM.value, "", claim.text)
        if query_id in seen_query_ids:
            continue
        seen_query_ids.add(query_id)
        queries.append(
            RetrievalQuery(
                query_id=query_id,
                text=claim.text,
                atomic_claim_id=claim.claim_id,
                question_id=None,
                origin=QueryOrigin.CLAIM,
            )
        )

    for question in probe_questions:
        query_id = make_query_id(question.atomic_claim_id, QueryOrigin.QUESTION.value, question.question_id, question.text)
        if query_id in seen_query_ids:
            continue
        seen_query_ids.add(query_id)
        queries.append(
            RetrievalQuery(
                query_id=query_id,
                text=question.text,
                atomic_claim_id=question.atomic_claim_id,
                question_id=question.question_id,
                origin=QueryOrigin.QUESTION,
            )
        )

    return queries
