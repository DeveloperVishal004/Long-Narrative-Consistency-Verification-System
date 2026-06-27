"""CrossEncoderNLIModel tests: label-map construction (offline) and real-model inference (gated)."""

import pytest

from lncvs.reasoning.nli.config import NLIConfig
from lncvs.reasoning.nli.model import CrossEncoderNLIModel, _build_label_map
from lncvs.schemas import NLILabel


def test_build_label_map_handles_any_id2label_ordering() -> None:
    """The label map must be derived from the model's own id2label, not assumed.

    This is the direct mitigation for the highest-risk Phase 5 failure mode:
    a checkpoint that orders its output classes differently than expected.
    """
    id2label_contradiction_first = {0: "contradiction", 1: "entailment", 2: "neutral"}
    id2label_entailment_first = {0: "entailment", 1: "neutral", 2: "contradiction"}

    map_a = _build_label_map(id2label_contradiction_first)
    map_b = _build_label_map(id2label_entailment_first)

    assert map_a[0] is NLILabel.CONTRADICTION
    assert map_a[1] is NLILabel.ENTAILMENT
    assert map_b[0] is NLILabel.ENTAILMENT
    assert map_b[2] is NLILabel.CONTRADICTION


def test_build_label_map_is_case_and_whitespace_insensitive() -> None:
    id2label = {0: " Contradiction ", 1: "ENTAILMENT", 2: "Neutral"}
    label_map = _build_label_map(id2label)
    assert label_map == {0: NLILabel.CONTRADICTION, 1: NLILabel.ENTAILMENT, 2: NLILabel.NEUTRAL}


def test_build_label_map_rejects_unrecognized_label() -> None:
    id2label = {0: "contradiction", 1: "entailment", 2: "something_else"}
    with pytest.raises(ValueError, match="Unrecognized NLI label"):
        _build_label_map(id2label)


@pytest.fixture(scope="module")
def real_nli_model() -> CrossEncoderNLIModel:
    config = NLIConfig(model_name="cross-encoder/nli-deberta-v3-base")
    try:
        return CrossEncoderNLIModel(config)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Could not load NLI model in this environment: {exc}")


def test_real_model_detects_contradiction(real_nli_model: CrossEncoderNLIModel) -> None:
    prediction = real_nli_model.predict(
        premise="John lost his left arm in an accident in 2010.",
        hypothesis="John used both hands.",
    )
    assert prediction.label is NLILabel.CONTRADICTION
    assert prediction.score > 0.5


def test_real_model_detects_entailment(real_nli_model: CrossEncoderNLIModel) -> None:
    prediction = real_nli_model.predict(
        premise="John moved to London in 2012.",
        hypothesis="The event occurred in London.",
    )
    assert prediction.label is NLILabel.ENTAILMENT
    assert prediction.score > 0.5
