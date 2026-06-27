"""Deterministic, offline test double for the NLIModel protocol."""

from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.schemas import NLILabel


class FakeNLIModel:
    """A scripted NLIModel: returns a fixed prediction for known (premise, hypothesis)
    pairs, or a default prediction for any pair if one is configured. Records
    every call, so tests can assert call counts (e.g. to prove a
    CachingNLIModel avoided a redundant call)."""

    def __init__(
        self,
        scripted: dict[tuple[str, str], NLIPrediction] | None = None,
        default_prediction: NLIPrediction | None = None,
    ) -> None:
        self._scripted = scripted or {}
        self._default_prediction = default_prediction or NLIPrediction(label=NLILabel.NEUTRAL, score=0.5)
        self.calls: list[tuple[str, str]] = []

    def predict(self, premise: str, hypothesis: str) -> NLIPrediction:
        self.calls.append((premise, hypothesis))
        return self._scripted.get((premise, hypothesis), self._default_prediction)
