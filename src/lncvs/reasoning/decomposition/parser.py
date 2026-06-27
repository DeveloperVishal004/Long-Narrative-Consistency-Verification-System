"""Pure parser: raw LLM completion text -> list[AtomicClaim].

No LLM calls happen here. Deterministic given identical input, which is
what makes this module fast and exhaustively unit-testable offline.
"""

import json
import re

from lncvs.reasoning.decomposition.identity import make_atomic_claim_id
from lncvs.schemas import AtomicClaim

# Real-execution finding (Phase H1, gemini-2.5-flash): despite the prompt's
# "Output ONLY a JSON array of strings, nothing else" instruction, ~29% of
# real decomposition calls wrapped the array in a markdown code fence
# (```json ... ```), which a strict json.loads() cannot parse -- it fails
# on the leading backtick with "Expecting value: line 1 column 1 (char 0)".
# This regex strips one optional leading ```/```json and trailing ``` if
# present; absent a fence, the text is returned unchanged, so this is a
# pure no-op for the common, correctly-unwrapped case.
_MARKDOWN_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_markdown_fence(text: str) -> str:
    match = _MARKDOWN_FENCE_PATTERN.match(text.strip())
    return match.group(1) if match else text


def parse_decomposition_response(
    raw_text: str, parent_claim_id: str, max_atomic_claims: int
) -> list[AtomicClaim]:
    """Parse a decomposition completion into a deduplicated, ordered list of AtomicClaims.

    Raises ValueError on malformed JSON, a non-array response, a non-string
    array element, zero resulting claims, or exceeding max_atomic_claims.
    Never returns an empty list — decomposition must yield at least one
    atomic claim, or the caller should treat this as a failure, not a
    vacuous success.
    """
    try:
        data = json.loads(_strip_markdown_fence(raw_text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Decomposition response is not valid JSON: {exc}") from exc

    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("Decomposition response must be a JSON array of strings")

    deduped_texts: list[str] = []
    seen: set[str] = set()
    for item in data:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_texts.append(normalized)

    if not deduped_texts:
        raise ValueError("Decomposition produced zero atomic claims")

    if len(deduped_texts) > max_atomic_claims:
        raise ValueError(
            f"Decomposition produced {len(deduped_texts)} atomic claims, "
            f"exceeding max_atomic_claims={max_atomic_claims}"
        )

    return [
        AtomicClaim(
            claim_id=make_atomic_claim_id(parent_claim_id, index, text),
            text=text,
            parent_claim_id=parent_claim_id,
            index=index,
        )
        for index, text in enumerate(deduped_texts)
    ]
