"""Retrieval: the Retriever protocol, semantic + lexical implementations, and orchestration."""

from lncvs.retrieval.base import Retriever
from lncvs.retrieval.bm25 import BM25Retriever
from lncvs.retrieval.config import RetrievalConfig
from lncvs.retrieval.grouping import group_evidence_by_claim
from lncvs.retrieval.identity import make_evidence_id, make_query_id
from lncvs.retrieval.orchestrator import RetrievalOrchestrator
from lncvs.retrieval.query_builder import build_retrieval_queries
from lncvs.retrieval.semantic import SemanticRetriever

__all__ = [
    "BM25Retriever",
    "RetrievalConfig",
    "RetrievalOrchestrator",
    "Retriever",
    "SemanticRetriever",
    "build_retrieval_queries",
    "group_evidence_by_claim",
    "make_evidence_id",
    "make_query_id",
]
