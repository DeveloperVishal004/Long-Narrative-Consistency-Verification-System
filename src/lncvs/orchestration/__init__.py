"""LangGraph Integration: ports the linear pipeline (Phases 0-5) onto a LangGraph StateGraph.

Model A (approved Phase 7 architecture): GraphState stays {ledger, control},
unchanged from schemas.state. EvidenceLedger remains the single source of
truth, mutated only through LedgerService -- the graph adds no second
mutation path. No checkpointer, no streaming, no parallel fan-out: this is
a strictly linear port, not a redesign.

orchestration/ must never import from evaluation/ -- the canonical
dependency chain is `... -> rules -> orchestration -> evaluation`.
PipelineRunner (evaluation/runner.py) is retained as the permanent
equivalence oracle; see tests/orchestration/test_graph_equivalence.py.
"""

from lncvs.orchestration.fusion_baselines import round_robin_fuse
from lncvs.orchestration.graph import LangGraphPipeline, build_graph
from lncvs.orchestration.resources import PipelineResources, RunContext

__all__ = ["LangGraphPipeline", "PipelineResources", "RunContext", "build_graph", "round_robin_fuse"]
