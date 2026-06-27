"""round_robin_fuse moved to orchestration/fusion_baselines.py in Phase 7
(both PipelineRunner and the LangGraph fuse node must call the identical
function, and orchestration/ must not import from evaluation/). Re-exported
here so every existing `from lncvs.evaluation import round_robin_fuse` or
`from lncvs.evaluation.fusion_baselines import round_robin_fuse` import
keeps working unchanged.
"""

from lncvs.orchestration.fusion_baselines import round_robin_fuse

__all__ = ["round_robin_fuse"]
