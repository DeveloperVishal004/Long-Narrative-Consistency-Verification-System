"""Deterministic, rule-based entity-mention extraction.

Stage G1 explicitly uses no NLP/NER model -- per the Phase 8 architecture
review, "zero models" is the defining property of this stage. The proxy
signal is capitalized-token extraction: a run of one or more consecutive
capitalized words (so "New York"-style multi-word names merge into a
single mention) with a small, fixed stopword list filtering out common
capitalized non-entity words (sentence-initial "The", pronouns, etc).

This is deliberately weak and over-inclusive compared to real NER -- see
EntityType's docstring for why entity_type is never guessed from this
signal. extract_mentions() is the single function used on both the corpus
side (graph construction) and the query side (entry-node resolution), the
same shared-tokenization discipline lncvs.indexing.tokenizer enforces for
BM25: divergence between the two would be a silent recall killer here too.
"""

import re

_WORD_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]*\b")

_STOPWORDS = frozenset(
    {
        "The", "A", "An", "This", "That", "These", "Those", "It", "He", "She",
        "They", "We", "I", "In", "On", "At", "Is", "Are", "Was", "Were", "His",
        "Her", "Their", "Its", "My", "Your", "Our", "If", "But", "And", "Or",
        "So", "Yet", "As", "Of", "To", "For", "With", "From", "By", "Then",
    }
)


def extract_mentions(text: str, min_token_length: int) -> list[str]:
    """Return the ordered, deduplicated list of capitalized-token mentions in text.

    Each individual capitalized word is checked against the stopword list
    *before* merging into a multi-word mention -- a stopword always breaks
    a run, so "Then John" can never merge into one mention just because
    both words are capitalized and adjacent. Two non-stopword capitalized
    words merge into a single multi-word mention only if they are
    separated by nothing but whitespace in the original text (so "New
    York" merges, but "New found York" -- two words with text between
    them -- does not).

    A mention shorter than min_token_length characters is dropped. Order
    of first appearance is preserved; duplicates within the same text are
    removed.
    """
    mentions: dict[str, None] = {}
    current_words: list[str] = []
    previous_end: int | None = None

    def flush() -> None:
        if not current_words:
            return
        phrase = " ".join(current_words)
        if len(phrase) >= min_token_length:
            mentions.setdefault(phrase, None)
        current_words.clear()

    for match in _WORD_PATTERN.finditer(text):
        word = match.group(0)
        is_adjacent = previous_end is not None and text[previous_end : match.start()].strip() == ""

        if word in _STOPWORDS:
            flush()
        else:
            if current_words and not is_adjacent:
                flush()
            current_words.append(word)

        previous_end = match.end()

    flush()
    return list(mentions.keys())
