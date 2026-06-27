"""PipelineResources and RunContext: dependency injection for LangGraph nodes.

Neither is a Pydantic model and neither ever enters GraphState. They hold
non-serializable objects (an Embedder, an NLIModel, LLMClients, retriever
wrappers) that are not domain state -- the same discipline as "ledger
stores chunk IDs, not chunk bodies" applied to the orchestration layer.
Both travel through LangGraph exclusively via `config["configurable"]`.
"""

from dataclasses import dataclass

from lncvs.chunking import ChunkingConfig
from lncvs.fusion import FusionConfig
from lncvs.indexing import Embedder
from lncvs.llm import LLMClient
from lncvs.reasoning.decomposition import DecompositionConfig
from lncvs.reasoning.nli import NLIModel
from lncvs.reasoning.questions import QuestionGenerationConfig
from lncvs.retrieval import Retriever
from lncvs.rules import RuleEngineConfig
from lncvs.schemas import AblationVariant


@dataclass(frozen=True)
class PipelineResources:
    """Graph-level dependencies, injected once per LangGraphPipeline instance
    and shared across every run() call.

    Mirrors exactly the dependency set lncvs.evaluation.runner.PipelineRunner
    takes at construction -- this parity is what makes the graph and the
    runner equivalent: both wrap the same injected models/configs around the
    same underlying service calls.
    """

    embedder: Embedder
    nli_model: NLIModel
    decomposition_llm: LLMClient
    question_llm: LLMClient
    decomposition_config: DecompositionConfig
    question_config: QuestionGenerationConfig
    rule_config: RuleEngineConfig
    chunking_config: ChunkingConfig
    retrieval_top_k: int = 10
    fusion_config: FusionConfig | None = None

    def fusion_config_or_default(self) -> FusionConfig:
        return self.fusion_config or FusionConfig()


@dataclass
class RunContext:
    """Mutable, per-run() holder for the active variant and the retrievers
    built by the ingest_and_index node.

    Lifetime-scoped to exactly one LangGraphPipeline.run() call -- the
    direct equivalent of the local `chroma_index`/`bm25_index`/`retrievers`
    variables PipelineRunner.run() builds on its call stack, just threaded
    through `configurable` across graph nodes instead of a single function's
    locals. Never a Pydantic model and never part of GraphState.
    """

    variant: AblationVariant
    semantic_retriever: Retriever | None = None
    lexical_retriever: Retriever | None = None
