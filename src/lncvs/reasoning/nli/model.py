"""NLIModel — the cross-encoder model abstraction and its prediction DTO.

NLIPrediction is intentionally NOT a schemas/ type: it never crosses out of
reasoning/nli/ (this file -> service.py, the same subpackage). NLIResult,
which IS the schemas/ type, is what crosses the module boundary once
NLIVerifier attaches claim/evidence provenance to a prediction.

CrossEncoderNLIModel is the only file in this codebase that imports
sentence_transformers.CrossEncoder, mirroring the chromadb/rank_bm25
isolation discipline elsewhere.
"""

import logging
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from lncvs.reasoning.nli.config import NLIConfig
from lncvs.schemas import NLILabel

logger = logging.getLogger(__name__)

_LABEL_ALIASES: dict[str, NLILabel] = {
    "contradiction": NLILabel.CONTRADICTION,
    "entailment": NLILabel.ENTAILMENT,
    "neutral": NLILabel.NEUTRAL,
}


class NLIPrediction(BaseModel):
    """A single NLI prediction: the argmax label and its softmax confidence."""

    model_config = ConfigDict(frozen=True)

    label: NLILabel = Field(..., description="ENTAILMENT, CONTRADICTION, or NEUTRAL.")
    score: float = Field(..., ge=0.0, le=1.0, description="Softmax confidence for the assigned label.")


@runtime_checkable
class NLIModel(Protocol):
    """Dependency-injection point for NLI inference.

    Any implementation (real cross-encoder, fake/deterministic test double)
    must satisfy this shape. NLIVerifier depends on this protocol, never on
    a concrete model library.
    """

    def predict(self, premise: str, hypothesis: str) -> NLIPrediction:
        """Return the NLI prediction for (premise, hypothesis)."""
        ...


def _build_label_map(id2label: dict[int, str]) -> dict[int, NLILabel]:
    """Build an index->NLILabel map from a model's own id2label, never a hardcoded guess.

    Different NLI checkpoints order their output classes differently
    (e.g. some put CONTRADICTION at index 0, others at index 1). Trusting a
    hardcoded index order is the single highest-risk silent failure mode in
    this module — it would invert every verdict without raising. Reading
    id2label directly from the loaded model's own config eliminates that
    class of bug entirely.
    """
    label_map: dict[int, NLILabel] = {}
    for index, raw_label in id2label.items():
        normalized = raw_label.strip().lower()
        if normalized not in _LABEL_ALIASES:
            raise ValueError(
                f"Unrecognized NLI label {raw_label!r} at index {index} in model id2label; "
                "cannot safely build a label map for this checkpoint."
            )
        label_map[index] = _LABEL_ALIASES[normalized]
    return label_map


class CrossEncoderNLIModel:
    """NLIModel backed by a sentence-transformers CrossEncoder.

    The model is loaded once, eagerly, at construction time. The label map
    is derived from the model's own id2label config (see _build_label_map)
    rather than assumed, so a checkpoint with a different class ordering
    fails loudly at construction instead of silently inverting verdicts.
    """

    def __init__(self, config: NLIConfig) -> None:
        from sentence_transformers import CrossEncoder

        self._config = config
        logger.info("Loading NLI model %r on device %r", config.model_name, config.device)
        self._model = CrossEncoder(
            config.model_name, max_length=config.max_length, device=config.device
        )
        self._label_map = _build_label_map(self._model.model.config.id2label)

    @property
    def tokenizer(self):
        """The underlying HuggingFace tokenizer, exposed read-only for
        tooling (e.g. token-budget analysis, truncation auditing) that
        needs to inspect how text will be tokenized without duplicating
        the CrossEncoder's own tokenization logic. Never used by predict()
        itself -- this is introspection, not a second inference path."""
        return self._model.tokenizer

    def predict(self, premise: str, hypothesis: str) -> NLIPrediction:
        raw_scores = self._model.predict([(premise, hypothesis)])[0]
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probabilities = exp_scores / exp_scores.sum()

        best_index = int(np.argmax(probabilities))
        return NLIPrediction(label=self._label_map[best_index], score=float(probabilities[best_index]))
