"""CachingNLIModel determinism and cache-hit tests, mirroring tests/llm and tests/indexing cache tests."""

from lncvs.reasoning.nli import CachingNLIModel, InMemoryNLICache, NLIConfig
from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.schemas import NLILabel
from tests.reasoning.nli.fakes import FakeNLIModel


def test_caching_model_avoids_redundant_calls_for_identical_input() -> None:
    fake = FakeNLIModel(
        default_prediction=NLIPrediction(label=NLILabel.CONTRADICTION, score=0.9)
    )
    config = NLIConfig(model_name="fake-model")
    caching_model = CachingNLIModel(fake, InMemoryNLICache(), config)

    first = caching_model.predict("premise text", "hypothesis text")
    second = caching_model.predict("premise text", "hypothesis text")

    assert first == second
    assert len(fake.calls) == 1


def test_caching_model_namespaces_by_config_fingerprint() -> None:
    fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.6))
    shared_cache = InMemoryNLICache()

    model_a = CachingNLIModel(fake, shared_cache, NLIConfig(model_name="model-a"))
    model_b = CachingNLIModel(fake, shared_cache, NLIConfig(model_name="model-b"))

    model_a.predict("premise", "hypothesis")
    model_b.predict("premise", "hypothesis")

    assert len(fake.calls) == 2


def test_caching_model_is_deterministic_across_repeated_runs() -> None:
    fake = FakeNLIModel(default_prediction=NLIPrediction(label=NLILabel.ENTAILMENT, score=0.95))
    config = NLIConfig(model_name="fake-model")

    def run_once() -> NLIPrediction:
        caching_model = CachingNLIModel(fake, InMemoryNLICache(), config)
        return caching_model.predict("premise", "hypothesis")

    assert run_once() == run_once()
