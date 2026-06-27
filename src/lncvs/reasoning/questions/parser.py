"""Pure parser: raw LLM completion text -> list[ProbeQuestion].

No LLM calls happen here. Deterministic given identical input.

Unlike claim decomposition's parser, an empty result is a VALID outcome —
a claim with no useful probe question is legitimate, since retrieval can
always fall back to the claim text itself. Malformed output (bad JSON,
wrong shape) still raises; only a genuinely empty *valid* array (or an
array reduced to empty by filtering) returns [].
"""

import json

from lncvs.reasoning.questions.identity import make_question_id
from lncvs.schemas import ProbeQuestion


def parse_question_response(
    raw_text: str, atomic_claim_id: str, max_questions_per_claim: int
) -> list[ProbeQuestion]:
    """Parse a question-generation completion into a deduplicated, ordered list of ProbeQuestions.

    Raises ValueError on malformed JSON, a non-array response, a non-string
    array element, or exceeding max_questions_per_claim. Returns [] if the
    model legitimately produced no questions, or if every candidate was
    filtered out for not being phrased as a question (see _looks_like_question) —
    this is a structural proxy enforcing that probe questions ask rather
    than assert a new, unstated fact as true.
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Question generation response is not valid JSON: {exc}") from exc

    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("Question generation response must be a JSON array of strings")

    deduped_texts: list[str] = []
    seen: set[str] = set()
    for item in data:
        normalized = item.strip()
        if not normalized or not _looks_like_question(normalized):
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_texts.append(normalized)

    if len(deduped_texts) > max_questions_per_claim:
        raise ValueError(
            f"Question generation produced {len(deduped_texts)} questions, "
            f"exceeding max_questions_per_claim={max_questions_per_claim}"
        )

    return [
        ProbeQuestion(
            question_id=make_question_id(atomic_claim_id, index, text),
            atomic_claim_id=atomic_claim_id,
            text=text,
            index=index,
        )
        for index, text in enumerate(deduped_texts)
    ]


def _looks_like_question(text: str) -> bool:
    """Structural proxy for "this asks, rather than asserts a new fact as true."

    Not a substitute for true semantic faithfulness checking (deferred to an
    LLM-judge evaluation in a later phase) — only catches the clearest
    failure mode: a declarative statement returned instead of a question.
    """
    return text.endswith("?")
