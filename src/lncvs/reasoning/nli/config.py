"""NLI cross-encoder model configuration."""

import hashlib

from pydantic import BaseModel, ConfigDict, Field


class NLIConfig(BaseModel):
    """Configures which NLI cross-encoder a NLIModel implementation loads and how it runs.

    Kept as configuration rather than a hard-coded constant so the model can
    be swapped (e.g. for evaluation) without code changes, mirroring
    EmbeddingConfig and LLMConfig.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str = Field(..., min_length=1, description="sentence-transformers CrossEncoder model identifier.")
    max_length: int = Field(default=256, gt=0, description="Maximum token length for premise+hypothesis input.")
    device: str = Field(default="cpu", description="Device to run NLI inference on.")

    def fingerprint(self) -> str:
        """Deterministic identifier for this exact configuration.

        Used to namespace cached predictions so a config change (different
        model_name, max_length, or device) can never silently serve a
        prediction computed under a different configuration.
        """
        digest_input = f"{self.model_name}:{self.max_length}:{self.device}".encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()[:16]
