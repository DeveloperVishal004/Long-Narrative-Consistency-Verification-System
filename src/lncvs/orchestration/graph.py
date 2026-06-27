"""build_graph() and LangGraphPipeline: the compiled Phase 7 StateGraph and
its PipelineRunner-equivalent run() interface.

No checkpointer is configured (graph.compile() with no checkpointer
argument): this keeps state semantics simple and deterministic. LedgerService
mutates EvidenceLedger in place and nodes return that same object reference
as the channel update -- harmless only because there is no
serialize/restore cycle that could ever observe a stale copy. Adding a
checkpointer later requires re-reviewing that in-place mutation first (see
Phase 7 architecture review Risk 1).
"""

import hashlib
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from lncvs.orchestration.nodes import (
    decompose_claim,
    error_sink,
    fuse,
    generate_questions,
    ingest_and_index,
    retrieve,
    route_after,
    verdict,
    verify_nli,
)
from lncvs.orchestration.resources import PipelineResources, RunContext
from lncvs.orchestration.state_channels import GraphChannels
from lncvs.chunking import ChunkingConfig
from lncvs.schemas import AblationVariant, ControlState, EvidenceLedger, PipelineStage


def build_graph():
    """Assemble and compile the linear, 8-node Phase 7 StateGraph.

    Edges are an unconditional linear happy path plus one conditional error
    edge per node, routing to a shared error_sink. Ablation (use_bm25,
    use_question_generation, fusion_strategy) is handled INSIDE nodes via
    the same inline conditionals PipelineRunner uses -- never via conditional
    edges -- which is what keeps the graph a faithful port rather than a
    second place ablation logic could drift from the oracle.
    """
    graph = StateGraph(GraphChannels)

    graph.add_node("ingest_and_index", ingest_and_index)
    graph.add_node("decompose_claim", decompose_claim)
    graph.add_node("generate_questions", generate_questions)
    graph.add_node("retrieve", retrieve)
    graph.add_node("fuse", fuse)
    graph.add_node("verify_nli", verify_nli)
    graph.add_node("verdict", verdict)
    graph.add_node("error_sink", error_sink)

    graph.add_edge(START, "ingest_and_index")
    graph.add_conditional_edges(
        "ingest_and_index", route_after("decompose_claim"), {"decompose_claim": "decompose_claim", "error_sink": "error_sink"}
    )
    graph.add_conditional_edges(
        "decompose_claim",
        route_after("generate_questions"),
        {"generate_questions": "generate_questions", "error_sink": "error_sink"},
    )
    graph.add_conditional_edges(
        "generate_questions", route_after("retrieve"), {"retrieve": "retrieve", "error_sink": "error_sink"}
    )
    graph.add_conditional_edges("retrieve", route_after("fuse"), {"fuse": "fuse", "error_sink": "error_sink"})
    graph.add_conditional_edges(
        "fuse", route_after("verify_nli"), {"verify_nli": "verify_nli", "error_sink": "error_sink"}
    )
    graph.add_conditional_edges(
        "verify_nli", route_after("verdict"), {"verdict": "verdict", "error_sink": "error_sink"}
    )
    graph.add_edge("verdict", END)
    graph.add_edge("error_sink", END)

    return graph.compile()


class LangGraphPipeline:
    """PipelineRunner-equivalent runner backed by a compiled LangGraph StateGraph.

    Satisfies lncvs.evaluation.runner.LedgerProducer structurally (same
    chunking_config property, same run() signature) without either module
    importing the other's concrete type.
    """

    def __init__(self, resources: PipelineResources) -> None:
        self._resources = resources
        self._compiled = build_graph()

    @property
    def chunking_config(self) -> ChunkingConfig:
        return self._resources.chunking_config

    def run(self, narrative_path: Path, original_claim: str, variant: AblationVariant) -> EvidenceLedger:
        run_context = RunContext(variant=variant)
        initial_state = GraphChannels(
            ledger=EvidenceLedger(original_claim=original_claim),
            control=ControlState(
                current_stage=PipelineStage.INGESTION,
                config_fingerprint=self._config_fingerprint(variant),
            ),
        )

        final_state = self._compiled.invoke(
            initial_state,
            config={
                "configurable": {
                    "resources": self._resources,
                    "run_context": run_context,
                    "narrative_path": narrative_path,
                }
            },
        )

        ledger: EvidenceLedger = final_state["ledger"]
        control: ControlState = final_state["control"]

        if control.current_stage is PipelineStage.ERROR:
            error_messages = "; ".join(f"{error.stage.value}: {error.message}" for error in control.errors)
            raise RuntimeError(f"LangGraphPipeline failed: {error_messages}")
        if ledger.final_verdict is None:
            raise RuntimeError("LangGraphPipeline completed without producing a final_verdict")

        return ledger

    def _config_fingerprint(self, variant: AblationVariant) -> str:
        """Orchestration-only provenance metadata for ControlState. Never
        enters the ledger, so it is equivalence-neutral by construction --
        intentionally best-effort, not an exhaustive hash of every config."""
        parts = [
            self._resources.decomposition_config.llm_config.fingerprint(),
            self._resources.question_config.llm_config.fingerprint(),
            variant.fingerprint(),
        ]
        return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()[:16]
