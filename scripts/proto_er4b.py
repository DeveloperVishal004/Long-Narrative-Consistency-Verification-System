"""ER4 prototype v2 (READ-ONLY): targeted surname-ambiguity detection.

Key insight from v1: classifying EVERY single-token key as "weak" shatters
ordinary characters (Paganel, Glenarvan, MacNabb) who are simply always
referred to by bare surname -- that is the normal case, not ambiguity.
The real signal for "this surname is a family collision" is: do MULTIPLE
DISTINCT qualifiers (a given name, or a gender-contrasting honorific like
Lord/Lady, M./Madame/Mademoiselle) appear attached to the same trailing
surname token across the corpus? If yes (Danglars: Baron+Eugénie+Madame;
Glenarvan: Lord+Lady; Villefort: M.+Madame+Mademoiselle) -> ambiguous,
needs corroboration. If no (Wilmore: always "Lord", never another title
or given name; MacNabb: always "Major") -> safe, treat as strong.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from evaluate_dataset import BOOK_NAME_TO_PATH  # noqa: E402
from audit_er3_merge_keys import NOBILIARY_PARTICLES  # noqa: E402

from lncvs.chunking import ChunkingConfig, chunk_document  # noqa: E402
from lncvs.graph.entity_resolution.normalization import is_generic_referent, norm_name  # noqa: E402
from lncvs.graph.llm_extraction import ExtractionConfig, LLMWindowExtractor  # noqa: E402
from lncvs.graph.provenance.service import resolve_window_provenance  # noqa: E402
from lncvs.graph.segmentation import segment_into_extraction_windows  # noqa: E402
from lncvs.ingestion import load_and_clean_narrative  # noqa: E402
from lncvs.llm import CachingStructuredLLMClient, JsonlStructuredLLMCache, LLMConfig  # noqa: E402

import logging

logging.disable(logging.WARNING)
RESULTS = REPO_ROOT / "results"

MALE_SENIOR_MARKERS = {"lord", "sir", "baron", "count", "duke", "prince", "king", "father"}
MALE_JUNIOR_MARKERS = {"vicomte", "viscount"}
MALE_NEUTRAL_MARKERS = {"mr", "monsieur", "m", "captain", "major", "general", "colonel", "abbe", "abbé", "dr"}
FEMALE_MARRIED_MARKERS = {"lady", "mrs", "madame", "mme", "baroness", "countess",
                           "duchess", "princess", "mother", "queen"}
FEMALE_UNMARRIED_MARKERS = {"miss", "mademoiselle", "mlle"}


class _NoCall:
    def complete_structured(self, prompt, schema):
        raise ValueError("cache-only")


def load_raw_mentions(book, cache_file, src_id):
    """Returns list of dicts: {primary, aliases(raw strings), etype, idx}"""
    document = load_and_clean_narrative(Path(BOOK_NAME_TO_PATH[book]), source_id=src_id)
    chunks = chunk_document(document, ChunkingConfig(chunk_size=700, overlap=120))
    windows = segment_into_extraction_windows(document.cleaned_text)
    ec = ExtractionConfig()
    cache = JsonlStructuredLLMCache(RESULTS / cache_file)
    extractor = LLMWindowExtractor(
        CachingStructuredLLMClient(_NoCall(), cache, LLMConfig(model_name="gemini-2.5-flash", temperature=0.0, max_tokens=65536), ec.schema_version), ec
    )
    facts = []
    for w in windows:
        wt = document.cleaned_text[w.char_start : w.char_end]
        try:
            ext = extractor.extract(wt, w.chapter_index, w.window_index)
        except ValueError:
            continue
        res = resolve_window_provenance(ext, wt, w.chapter_index, w.window_index, w.char_start, chunks)
        facts.extend(res.resolved_entities)

    import hashlib

    def sk(f):
        return hashlib.sha256(f"{f.chapter_index}:{f.window_index}:{f.raw.local_id}:{f.raw.name}".encode()).hexdigest()

    facts.sort(key=sk)
    out = []
    for i, f in enumerate(facts):
        out.append({"idx": i, "primary": f.raw.name, "aliases": list(f.raw.aliases), "etype": str(f.raw.type).split(".")[-1]})
    return out


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def tokenize_raw(s: str) -> list[str]:
    return _PUNCT_RE.sub(" ", s.lower()).split()


def is_weak_key(key: str) -> bool:
    toks = key.split()
    if len(toks) == 1:
        return True
    return toks[0] in NOBILIARY_PARTICLES


def qualifier_tag(raw_surface: str, key_tokens: list[str]) -> str | None:
    """Given a raw surface form (e.g. 'Baron Danglars') and the surname
    key tokens it reduces to (e.g. ['danglars']), return a tag describing
    the DISTINGUISHING qualifier attached, or None if no distinguishing
    signal (bare name, or a qualifier we don't recognize)."""
    toks = tokenize_raw(raw_surface)
    # strip trailing occurrence of key_tokens if present at the end
    if len(toks) >= len(key_tokens) and toks[-len(key_tokens):] == key_tokens:
        prefix = toks[: -len(key_tokens)]
    else:
        prefix = [t for t in toks if t not in key_tokens]
    if not prefix:
        return None
    # Any prefix word that is neither a recognized title/honorific NOR a
    # generic referent (pronoun/determiner/etc, already excluded from
    # merge keys elsewhere) is treated as a generic "distinguishing
    # qualifier" -- deliberately NOT a hardcoded whitelist of fictional
    # characters' given names (which would overfit to this test corpus
    # and never generalize to a new novel). We don't need to know in
    # advance that "Eugenie" is a first name; we only need to know it
    # ISN'T a title, which is a closed, generalizable vocabulary.
    all_markers = (
        MALE_SENIOR_MARKERS | MALE_JUNIOR_MARKERS | MALE_NEUTRAL_MARKERS
        | FEMALE_MARRIED_MARKERS | FEMALE_UNMARRIED_MARKERS
    )
    other = sorted(t for t in prefix if t not in all_markers and not is_generic_referent(t))
    if other:
        return f"other:{' '.join(other)}"
    if any(t in MALE_JUNIOR_MARKERS for t in prefix):
        return "male_junior"
    if any(t in MALE_SENIOR_MARKERS for t in prefix):
        return "male_senior"
    if any(t in FEMALE_UNMARRIED_MARKERS for t in prefix):
        return "female_unmarried"
    if any(t in FEMALE_MARRIED_MARKERS for t in prefix):
        return "female_married"
    if any(t in MALE_NEUTRAL_MARKERS for t in prefix):
        return "male_neutral"
    return None


def best_tag_for_key(mention_keys, i, k, ktoks) -> str | None:
    """Try every surface that contributed key k for mention i (primary
    name first, then aliases in order) and return the first non-None tag
    found -- never just the last writer, which would silently discard a
    more specific (e.g. titled) surface in favor of a blander alias."""
    for surface in mention_keys[i][k]:
        tag = qualifier_tag(surface, ktoks)
        if tag is not None:
            return tag
    return None


def build(book, cache_file, src_id):
    raw = load_raw_mentions(book, cache_file, src_id)
    n = len(raw)

    # mention keys (post generic-filter + Clause A corroboration computed below)
    primary_norm = []
    for m in raw:
        pk = norm_name(m["primary"])
        primary_norm.append(pk if pk and not is_generic_referent(pk) else None)
    primary_keys_corpus = {k for k in primary_norm if k}

    mention_keys = []
    # key -> ALL raw surface strings (primary name first, then aliases) that
    # produced it for THIS mention. A bug fix from v1: a mention can list a
    # BARE alias ("Danglars") alongside a TITLED primary name ("Baron
    # Danglars") that both reduce to the same key -- storing only the last
    # writer silently lost the more specific (primary) surface, making a
    # genuinely-tagged mention look "bare". Storing all surfaces and trying
    # the most specific first (primary > aliases, in order) fixes this.
    for i, m in enumerate(raw):
        keys: dict[str, list[str]] = {}
        if primary_norm[i]:
            keys.setdefault(primary_norm[i], []).append(m["primary"])
        for a in m["aliases"]:
            ak = norm_name(a)
            if ak and not is_generic_referent(ak) and ak in primary_keys_corpus:
                keys.setdefault(ak, []).append(a)
        mention_keys.append(keys)

    key_to_idx = defaultdict(list)
    for i, keys in enumerate(mention_keys):
        for k in keys:
            key_to_idx[k].append(i)

    # ambiguity detection per weak key
    ambiguous_keys = set()
    weak_keys = {k for k in key_to_idx if is_weak_key(k)}
    for k in weak_keys:
        ktoks = k.split()
        tags = set()
        for i in key_to_idx[k]:
            tag = best_tag_for_key(mention_keys, i, k, ktoks)
            if tag:
                tags.add(tag)
        relationship_tags = {t for t in tags if not t.startswith("other:")}
        givens = {t for t in tags if t.startswith("other:")}
        if len(relationship_tags) >= 2 or len(givens) >= 2 or (relationship_tags and givens):
            ambiguous_keys.add(k)

    # Propagate ambiguity across the surname "core": a bare single-token
    # key ("Morcerf") never carries a qualifier prefix on its own, so it
    # can never trip the tag-diversity check above even though it is the
    # exact same family surname as an already-flagged particle-led key
    # ("de Morcerf", "of Morcerf"). Group weak keys by their trailing
    # surname core (the key itself for a single token; the key minus its
    # leading particle for a particle-led phrase) and mark every key in a
    # core-group ambiguous if ANY member of that group was flagged.
    core_groups = defaultdict(set)
    for k in weak_keys:
        toks = k.split()
        core = " ".join(toks[1:]) if len(toks) > 1 else k
        core_groups[core].add(k)
    for core, keys_in_group in core_groups.items():
        if keys_in_group & ambiguous_keys:
            ambiguous_keys |= keys_in_group

    return raw, mention_keys, key_to_idx, weak_keys, ambiguous_keys


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[max(ra, rb)] = min(ra, rb)
        return True


def resolve(raw, mention_keys, key_to_idx, weak_keys, ambiguous_keys):
    n = len(raw)
    uf = UF(n)
    safe_keys = set(key_to_idx) - ambiguous_keys  # strong + non-ambiguous weak: always union
    for k in safe_keys:
        idxs = key_to_idx[k]
        for o in idxs[1:]:
            uf.union(idxs[0], o)

    # ambiguous keys: a BARE (untagged) mention carries no distinguishing
    # signal, so it joins the majority ("anchor") tag-group for that key
    # freely. Mentions tagged with a DIFFERENT, conflicting qualifier (a
    # different given name, or the opposite gender marker) require a
    # second corroborating shared key -- checked against the SPECIFIC
    # mention pair's own (small, un-inflated) keysets, never against an
    # already-merged component's aggregated keys. Aggregated-component
    # corroboration is unsound: once a component absorbs many members it
    # accumulates enough unrelated keys that it "shares a key" with almost
    # anything, silently defeating the corroboration requirement.
    for k in ambiguous_keys:
        ktoks = k.split()
        idxs = key_to_idx[k]
        tag_of = {i: best_tag_for_key(mention_keys, i, k, ktoks) for i in idxs}
        tag_counts = defaultdict(int)
        for i in idxs:
            if tag_of[i] is not None:
                tag_counts[tag_of[i]] += 1
        anchor_tag = max(sorted(tag_counts), key=lambda t: tag_counts[t]) if tag_counts else None
        anchor_member = next((i for i in idxs if tag_of[i] is None or tag_of[i] == anchor_tag), idxs[0])

        for i in idxs:
            if tag_of[i] is None or tag_of[i] == anchor_tag:
                uf.union(anchor_member, i)
            else:
                # minority tag: union with any other mention sharing k that
                # ALSO shares a second key with it -- a real second piece
                # of corroborating evidence, checked pairwise (never
                # against an aggregated component's keyset).
                for j in idxs:
                    if j != i and (mention_keys[i].keys() & mention_keys[j].keys()) - {k}:
                        uf.union(i, j)

    comps = defaultdict(list)
    for i in range(n):
        comps[uf.find(i)].append(raw[i]["primary"])
    return list(comps.values()), uf


def name_to_comp_ids(comps):
    m = defaultdict(set)
    for idx, c in enumerate(comps):
        for nm in c:
            m[nm].add(idx)
    return m


def report(book, cache_file, src_id, preserve_pairs, family_pairs):
    raw, mention_keys, key_to_idx, weak_keys, ambiguous_keys = build(book, cache_file, src_id)
    print("=" * 78)
    print(book)
    print("=" * 78)
    print(f"mentions={len(raw)}  weak_keys={len(weak_keys)}  ambiguous_keys={len(ambiguous_keys)}")
    print(f"ambiguous keys found: {sorted(ambiguous_keys)}")
    comps, uf = resolve(raw, mention_keys, key_to_idx, weak_keys, ambiguous_keys)
    comps.sort(key=len, reverse=True)
    giant = comps[0]
    print(f"components={len(comps)}  giant_mentions={len(giant)}  giant_distinct_names={len(set(giant))}")
    n2c = name_to_comp_ids(comps)
    print("preserve checks:")
    for a, b in preserve_pairs:
        ok = bool(n2c.get(a, set()) & n2c.get(b, set()))
        print(f"   {a!r} <-> {b!r}: {'OK merged' if ok else '*** BROKEN ***'}")
    print("family-separation checks:")
    for a, b in family_pairs:
        same = bool(n2c.get(a, set()) & n2c.get(b, set()))
        print(f"   {a!r} vs {b!r}: {'STILL MERGED (bad)' if same else 'separated (good)'}")
    print(f"giant sample names: {sorted(set(giant))[:30]}")
    print()
    return comps


if __name__ == "__main__":
    report(
        "In Search of the Castaways", "extraction_cache_in_search_of_the_castaways.jsonl", "castaways",
        preserve_pairs=[
            ("Ayrton", "Ben Joyce"), ("Paganel", "Jacques Paganel"), ("Glenarvan", "Lord Glenarvan"),
            ("MacNabb", "Major MacNabb"), ("Captain Grant", "Harry Grant"),  # same person, not a collision
        ],
        family_pairs=[("Captain Grant", "Mary Grant"), ("Lord Glenarvan", "Lady Helena")],
    )
    report(
        "The Count of Monte Cristo", "extraction_cache_the_count_of_monte_cristo.jsonl", "mc",
        preserve_pairs=[
            ("Dantès", "Monte Cristo"), ("Dantès", "Edmond Dantès"), ("Lord Wilmore", "Dantès"),
            ("Abbé Busoni", "Dantès"), ("Sinbad the Sailor", "Dantès"),
        ],
        family_pairs=[
            ("Baron Danglars", "Eugénie Danglars"), ("M. de Villefort", "Valentine"),
            ("Andrea Cavalcanti", "Albert"),
        ],
    )
