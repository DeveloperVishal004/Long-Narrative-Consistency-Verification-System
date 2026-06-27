"""Test doubles specific to evaluation/ tests.

SubstringNLIModel exists because chunking is a sliding character window
(lncvs.chunking.chunker), not sentence-aware -- a chunk boundary can land
mid-sentence, so the chunk text a real retrieval pipeline hands to NLI is
rarely byte-identical to a hand-written sentence. The shared
tests.reasoning.nli.fakes.FakeNLIModel (exact (premise, hypothesis) dict
lookup) is the right contract for Phase 5's NLIVerifier unit tests, where
premise/hypothesis text is constructed directly by the test. Phase 6's
PipelineRunner tests instead drive real chunking, so a substring-match fake
is the correct level of robustness here: it survives surrounding text
without becoming a real (and non-deterministic) NLI model.
"""

from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.schemas import NLILabel


class SubstringNLIModel:
    """A scripted NLIModel: matches by substring containment in (premise, hypothesis),
    falling back to a default prediction. Deterministic and offline."""

    def __init__(
        self,
        rules: list[tuple[str, str, NLIPrediction]],
        default_prediction: NLIPrediction | None = None,
    ) -> None:
        self._rules = rules
        self._default_prediction = default_prediction or NLIPrediction(label=NLILabel.NEUTRAL, score=0.5)
        self.calls: list[tuple[str, str]] = []

    def predict(self, premise: str, hypothesis: str) -> NLIPrediction:
        self.calls.append((premise, hypothesis))
        for premise_substring, hypothesis_substring, prediction in self._rules:
            if premise_substring in premise and hypothesis_substring in hypothesis:
                return prediction
        return self._default_prediction
