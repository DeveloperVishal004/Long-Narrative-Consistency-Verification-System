"""RetrievalOrchestrator tests: provenance stamping and evidence_id collision-freedom."""

from lncvs.retrieval import RetrievalConfig, RetrievalOrchestrator, build_retrieval_queries
from lncvs.schemas import AtomicClaim, ProbeQuestion, QueryOrigin, RetrievalQuery, RetrievalSource
from tests.retrieval.fakes import FakeRetriever, make_unstamped_evidence


def test_stamps_atomic_claim_id_and_query_id_onto_results() -> None:
    query = RetrievalQuery(
        query_id="query-1", text="John used both hands", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM
    )
    retriever = FakeRetriever({"John used both hands": [make_unstamped_evidence("chunk-1", "evidence text")]})
    orchestrator = RetrievalOrchestrator([retriever], RetrievalConfig(top_k=5))

    results = orchestrator.retrieve_for_queries([query])

    assert len(results) == 1
    assert results[0].atomic_claim_id == "claim-1"
    assert results[0].query_id == "query-1"


def test_re_derives_evidence_id_from_query_id() -> None:
    query = RetrievalQuery(
        query_id="query-1", text="John used both hands", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM
    )
    unstamped = make_unstamped_evidence("chunk-1", "evidence text")
    retriever = FakeRetriever({"John used both hands": [unstamped]})
    orchestrator = RetrievalOrchestrator([retriever], RetrievalConfig(top_k=5))

    results = orchestrator.retrieve_for_queries([query])

    # evidence_id must no longer be the retriever's own raw-query-text derivation.
    assert results[0].evidence_id != unstamped.evidence_id


def test_two_claims_sharing_identical_query_text_do_not_collide() -> None:
    """The core correctness fix: a retriever's own evidence_id (derived from raw
    query text) would collide here; the orchestrator's re-derivation (via
    query_id, which encodes atomic_claim_id) must not."""
    shared_text = "Did John lose an arm?"
    query_for_claim_a = RetrievalQuery(
        query_id="query-a", text=shared_text, atomic_claim_id="claim-a", origin=QueryOrigin.CLAIM
    )
    query_for_claim_b = RetrievalQuery(
        query_id="query-b", text=shared_text, atomic_claim_id="claim-b", origin=QueryOrigin.CLAIM
    )
    # Same retriever call (same text) returns the identical underlying evidence twice.
    unstamped = make_unstamped_evidence("chunk-1", "evidence text")
    retriever = FakeRetriever({shared_text: [unstamped]})
    orchestrator = RetrievalOrchestrator([retriever], RetrievalConfig(top_k=5))

    results = orchestrator.retrieve_for_queries([query_for_claim_a, query_for_claim_b])

    assert len(results) == 2
    assert results[0].evidence_id != results[1].evidence_id
    assert results[0].atomic_claim_id == "claim-a"
    assert results[1].atomic_claim_id == "claim-b"


def test_zero_results_for_a_query_is_non_fatal() -> None:
    query = RetrievalQuery(
        query_id="query-1", text="an unmatched query", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM
    )
    retriever = FakeRetriever({"an unmatched query": []})
    orchestrator = RetrievalOrchestrator([retriever], RetrievalConfig(top_k=5))

    results = orchestrator.retrieve_for_queries([query])

    assert results == []


def test_passes_top_k_through_to_the_retriever() -> None:
    query = RetrievalQuery(
        query_id="query-1", text="John used both hands", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM
    )
    retriever = FakeRetriever({"John used both hands": [make_unstamped_evidence("chunk-1", "evidence text")]})
    orchestrator = RetrievalOrchestrator([retriever], RetrievalConfig(top_k=3))

    orchestrator.retrieve_for_queries([query])

    assert retriever.calls == [("John used both hands", 3)]


def test_results_are_deterministic_across_orchestrator_instances() -> None:
    query = RetrievalQuery(
        query_id="query-1", text="John used both hands", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM
    )

    def run_once() -> str:
        retriever = FakeRetriever({"John used both hands": [make_unstamped_evidence("chunk-1", "evidence text")]})
        orchestrator = RetrievalOrchestrator([retriever], RetrievalConfig(top_k=5))
        return orchestrator.retrieve_for_queries([query])[0].evidence_id

    assert run_once() == run_once()


def test_end_to_end_with_built_queries_across_multiple_atomic_claims() -> None:
    """Integration of query_builder + orchestrator, mirroring how a node would use both."""
    claim_piano = AtomicClaim(claim_id="claim-piano", text="John played piano", parent_claim_id="parent-1")
    claim_hands = AtomicClaim(claim_id="claim-hands", text="John used both hands", parent_claim_id="parent-1")
    question = ProbeQuestion(question_id="q-1", atomic_claim_id="claim-hands", text="Did John lose an arm?")

    queries = build_retrieval_queries([claim_piano, claim_hands], [question])

    retriever = FakeRetriever(
        {
            "John played piano": [make_unstamped_evidence("chunk-piano", "John played piano in a pub.")],
            "John used both hands": [],
            "Did John lose an arm?": [make_unstamped_evidence("chunk-arm", "John lost his left arm in 2010.")],
        }
    )
    orchestrator = RetrievalOrchestrator([retriever], RetrievalConfig(top_k=5))

    evidence = orchestrator.retrieve_for_queries(queries)

    by_claim = {}
    for item in evidence:
        by_claim.setdefault(item.atomic_claim_id, []).append(item)

    assert by_claim["claim-piano"][0].chunk_id == "chunk-piano"
    # The "John used both hands" claim query itself returns nothing, but its
    # probe question successfully surfaces the contradicting chunk.
    hands_evidence = [item for item in evidence if item.atomic_claim_id == "claim-hands"]
    assert len(hands_evidence) == 1
    assert hands_evidence[0].chunk_id == "chunk-arm"


# --- Multi-retriever (Phase 4): source-aware evidence_id collision-freedom ---


def test_runs_every_retriever_for_every_query() -> None:
    query = RetrievalQuery(
        query_id="query-1", text="John used both hands", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM
    )
    semantic_retriever = FakeRetriever(
        {"John used both hands": [make_unstamped_evidence("chunk-1", "evidence text", source=RetrievalSource.SEMANTIC)]}
    )
    lexical_retriever = FakeRetriever(
        {"John used both hands": [make_unstamped_evidence("chunk-1", "evidence text", source=RetrievalSource.LEXICAL)]}
    )
    orchestrator = RetrievalOrchestrator([semantic_retriever, lexical_retriever], RetrievalConfig(top_k=5))

    results = orchestrator.retrieve_for_queries([query])

    assert len(results) == 2
    assert {r.source for r in results} == {RetrievalSource.SEMANTIC, RetrievalSource.LEXICAL}


def test_same_query_chunk_and_rank_across_two_sources_do_not_collide() -> None:
    """The Phase 4 correctness fix: two different backends returning the same
    chunk at the same rank for the same query must not produce the same
    evidence_id — without source folded into the hash, they would."""
    query = RetrievalQuery(
        query_id="query-1", text="John used both hands", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM
    )
    semantic_retriever = FakeRetriever(
        {
            "John used both hands": [
                make_unstamped_evidence("chunk-arm", "John lost his left arm.", rank=1, source=RetrievalSource.SEMANTIC)
            ]
        }
    )
    lexical_retriever = FakeRetriever(
        {
            "John used both hands": [
                make_unstamped_evidence("chunk-arm", "John lost his left arm.", rank=1, source=RetrievalSource.LEXICAL)
            ]
        }
    )
    orchestrator = RetrievalOrchestrator([semantic_retriever, lexical_retriever], RetrievalConfig(top_k=5))

    results = orchestrator.retrieve_for_queries([query])

    assert len(results) == 2
    assert results[0].evidence_id != results[1].evidence_id
    assert results[0].chunk_id == results[1].chunk_id == "chunk-arm"


def test_single_retriever_list_behaves_as_phase_3_orchestrator_did() -> None:
    """A one-element retrievers list must produce identical behavior to the
    Phase 3 single-retriever orchestrator — no regression from the signature change."""
    query = RetrievalQuery(
        query_id="query-1", text="John used both hands", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM
    )
    retriever = FakeRetriever({"John used both hands": [make_unstamped_evidence("chunk-1", "evidence text")]})
    orchestrator = RetrievalOrchestrator([retriever], RetrievalConfig(top_k=5))

    results = orchestrator.retrieve_for_queries([query])

    assert len(results) == 1
    assert results[0].atomic_claim_id == "claim-1"
    assert results[0].query_id == "query-1"
