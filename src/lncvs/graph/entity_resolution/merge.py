"""Deterministic cross-window entity merging (frozen G2 spec §4, extended
by ER4's Clause A / Clause B corroboration rules -- see below).

Conservative policy, mandatory: merging is by name/alias equality only --
no fuzzy similarity, no embedding clustering. This deliberately biases
toward false splits over false merges -- a missed alias only reduces
multi-hop reach, while a false merge manufactures the wrong-character-
evidence failure mode the dataset evaluation already identified as this
project's dominant real-world error source. SAME_AS relations never
trigger a merge here -- they are an ordinary typed edge, handled entirely
in graph construction (Slice 6), never consulted by this module.

Generic-referent keys are excluded from the union step (is_generic_referent,
normalization.py): a normalized name that is entirely pronouns/bare
titles/kinship terms ("he", "the major", "my father") carries no entity
identity and must never bridge two mentions. Without it, the LLM
extractor's habit of listing such tokens as a mention's name or alias
transitively collapsed an entire novel's principal cast into one node on
the real corpus -- the dominant graph-retrieval failure this module exists
to prevent.

ER4 audit finding: even with generic referents excluded, two narrower
problems remained, both empirically confirmed on the real corpus before
any code changed (see the ER3/ER4 audit record):

Clause A (corroborated-alias rule): the LLM extractor also lists
DESCRIPTIVE EPITHETS as aliases ("the dying man", "my lord", "bandit",
"her lover") that are not pronouns/titles but still carry no entity
identity. An alias is only trustworthy as a merge key if its normalized
form ALSO appears, elsewhere in the corpus, as some mention's own PRIMARY
name -- genuine identity aliases ("Monte Cristo", "Lord Wilmore", "Ben
Joyce") satisfy this; invented epithets never do, because no entity is
ever introduced with "the dying man" as its name. A mention's own primary
name is never subject to this check -- only its ALIASES are alias claims
to be corroborated.

Clause B (weak-surname rule): a single-token surname ("Danglars") or a
particle-led surname ("de Villefort") is, on its own, WEAK evidence -- it
collapses an entire family (father, wife, daughter) into one node when the
text uses the bare surname for more than one family member. Two refinements
were required to implement this without breaking ordinary characters who
are simply always called by one bare name (Paganel, Glenarvan, MacNabb):

1. A weak key is only flagged AMBIGUOUS if the corpus actually shows two
   DIFFERENT distinguishing qualifiers attached to it -- a relationship-
   coded honorific contrast (Lord vs Lady, Baron vs Madame, M. vs
   Mademoiselle) or two different non-title qualifier words ("Eugénie
   Danglars" vs "Baron Danglars"). A surname used consistently with one
   title only ("Lord Wilmore", always) is NOT ambiguous and merges like
   any other key.
2. For an ambiguous key, a BARE mention (no qualifier at all -- the
   overwhelming majority of references) joins the corpus's most frequent
   ("anchor") qualifier group freely, since it carries no information that
   contradicts that default. Only a mention tagged with a MINORITY,
   conflicting qualifier requires a second corroborating shared key (a
   full name, a verified alias, another strong key) with some other
   mention sharing the surname before it is admitted -- checked against
   that SPECIFIC mention pair's own keys, never an already-merged
   component's aggregated keyset (which would accumulate enough unrelated
   keys to spuriously "corroborate" almost anything).

Distinguishing-qualifier detection deliberately does NOT use a hardcoded
whitelist of fictional characters' given names (which would overfit to
whatever novel happened to be tested and never generalize). Instead: any
qualifier word that is not a recognized title/honorific (a closed,
language-level vocabulary) and not itself a generic referent is treated as
a generic "distinguishing identifier" -- we don't need to know in advance
that "Eugénie" is a first name, only that it isn't a title.

Connected components are computed via union-find over mentions processed
in a fixed, content-hash-derived order (never raw list/dict iteration
order), so the same input always produces the same components regardless
of process or hash-seed.
"""

import hashlib
import re
from collections import defaultdict

from lncvs.graph.entity_resolution.normalization import is_generic_referent, norm_name
from lncvs.graph.identity import make_entity_id
from lncvs.graph.provenance.models import ResolvedFact
from lncvs.schemas import EntityRecord, EntityType

# Clause B: leading particles that mark a key as a (weak) surname phrase
# rather than a full name -- "de Villefort", "von Trapp", "of Morcerf".
_NOBILIARY_PARTICLES = frozenset({"de", "d", "du", "des", "la", "le", "von", "van", "of", "saint", "st"})

# Clause B relationship-marker vocabulary: closed, language-level title/
# honorific sets used ONLY to detect a genuine relationship contrast (e.g.
# Lord vs Lady = husband vs wife) attached to a shared surname. Deliberately
# not the same set as normalization._TITLE_PREFIXES, which strips titles
# unconditionally; here we need to know WHICH title was used, not erase it.
_MALE_SENIOR_MARKERS = frozenset({"lord", "sir", "baron", "count", "duke", "prince", "king", "father"})
_MALE_JUNIOR_MARKERS = frozenset({"vicomte", "viscount"})
_MALE_NEUTRAL_MARKERS = frozenset({"mr", "monsieur", "m", "captain", "major", "general", "colonel", "abbe", "abbé", "dr"})
_FEMALE_MARRIED_MARKERS = frozenset({"lady", "mrs", "madame", "mme", "baroness", "countess", "duchess", "princess", "mother", "queen"})
_FEMALE_UNMARRIED_MARKERS = frozenset({"miss", "mademoiselle", "mlle"})
_ALL_RELATIONSHIP_MARKERS = (
    _MALE_SENIOR_MARKERS | _MALE_JUNIOR_MARKERS | _MALE_NEUTRAL_MARKERS | _FEMALE_MARRIED_MARKERS | _FEMALE_UNMARRIED_MARKERS
)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _mention_sort_key(fact: ResolvedFact) -> str:
    """Deterministic processing order: content hash of the mention's
    window-scoped identity, never raw list/dict iteration order."""
    digest_input = f"{fact.chapter_index}:{fact.window_index}:{fact.raw.local_id}:{fact.raw.name}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def _find(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _union(parent: list[int], a: int, b: int) -> None:
    root_a, root_b = _find(parent, a), _find(parent, b)
    if root_a == root_b:
        return
    # Deterministic regardless of union order: the smaller index always
    # becomes the new root.
    if root_a < root_b:
        parent[root_b] = root_a
    else:
        parent[root_a] = root_b


def _is_weak_key(key: str) -> bool:
    """Clause B: a single-token key, or a multi-token key led by a
    nobiliary particle ("de Villefort"), is WEAK -- it unifies mentions
    only when not flagged ambiguous, or when ambiguous-but-corroborated
    (see _resolve_ambiguous_keys). A multi-token key with no leading
    particle ("Edmond Dantès", "Count of Monte Cristo", "Sinbad the
    Sailor") is a full name and is always STRONG."""
    tokens = key.split()
    return len(tokens) == 1 or tokens[0] in _NOBILIARY_PARTICLES


def _tokenize_raw(surface: str) -> list[str]:
    return _PUNCT_RE.sub(" ", surface.lower()).split()


def _qualifier_tag(raw_surface: str, key_tokens: list[str]) -> str | None:
    """Classify the qualifier attached to raw_surface beyond key_tokens
    (the bare surname/key it reduces to). Returns None for a bare mention
    (no qualifier at all). Returns "other:<word(s)>" for any qualifier word
    that is not a recognized title and not a generic referent -- this is
    the generalizable substitute for a given-name whitelist: we don't need
    to know "Eugénie" is a first name, only that it isn't a title. Returns
    a relationship-category tag (e.g. "male_senior", "female_married") for
    a qualifier composed entirely of recognized titles/honorifics.
    """
    tokens = _tokenize_raw(raw_surface)
    if len(tokens) >= len(key_tokens) and tokens[-len(key_tokens) :] == key_tokens:
        prefix = tokens[: -len(key_tokens)]
    else:
        prefix = [token for token in tokens if token not in key_tokens]
    if not prefix:
        return None

    distinguishing = sorted(token for token in prefix if token not in _ALL_RELATIONSHIP_MARKERS and not is_generic_referent(token))
    if distinguishing:
        return f"other:{' '.join(distinguishing)}"
    if any(token in _MALE_JUNIOR_MARKERS for token in prefix):
        return "male_junior"
    if any(token in _MALE_SENIOR_MARKERS for token in prefix):
        return "male_senior"
    if any(token in _FEMALE_UNMARRIED_MARKERS for token in prefix):
        return "female_unmarried"
    if any(token in _FEMALE_MARRIED_MARKERS for token in prefix):
        return "female_married"
    # Gender-neutral-ish formal titles ("M.", "Monsieur", "Captain") do NOT
    # get their own tag: they don't distinguish WHICH family member is
    # meant (any adult male in the family could be addressed this way), so
    # treating them as their own exclusive category would wrongly split
    # "M. Danglars" from "Baron Danglars" -- the same man. They are still
    # excluded from `distinguishing` above (so they're never mistaken for a
    # real identifying word); here they fall through to the same
    # no-distinguishing-signal outcome as a bare mention.
    return None


def _best_tag_for_key(mention_keys: list[dict[str, list[str]]], index: int, key: str, key_tokens: list[str]) -> str | None:
    """Try every surface that contributed key for this mention (primary
    name first, then aliases in encounter order) and return the first
    non-None tag found. Never just the last writer: a mention can list a
    bare alias ("Danglars") alongside a titled primary name ("Baron
    Danglars") that both reduce to the same key, and the more specific
    (titled) surface must not be silently discarded by whichever one
    happened to be recorded last."""
    for surface in mention_keys[index][key]:
        tag = _qualifier_tag(surface, key_tokens)
        if tag is not None:
            return tag
    return None


def _detect_ambiguous_keys(key_to_indices: dict[str, list[int]], mention_keys: list[dict[str, list[str]]]) -> set[str]:
    """A weak key is AMBIGUOUS only if the corpus shows real evidence of
    multiple distinct people sharing it: two different relationship-
    category tags (a gender/seniority contrast: Lord vs Lady, Baron vs
    Mademoiselle), two different "other:" distinguishing words, or one of
    each. A surname always used the same way (or always bare) is not
    ambiguous and merges unconditionally, like any other key.

    A bare single-token key ("Morcerf") sharing its surname with an
    ambiguous particle-led sibling ("de Morcerf") is deliberately NOT
    propagated to ambiguous here: given how _resolve_ambiguous_keys
    resolves an ambiguous key (every member tagged with the corpus's most
    common -- or no -- qualifier joins one "anchor" group freely), forcing
    a key whose own occupants show at most one distinct tag into the
    ambiguous path produces the identical grouping as leaving it
    unambiguous, confirmed empirically (real-corpus rebuild byte-identical
    with propagation removed) -- there is no pair of distinct identities
    to separate unless that key's own tags already show >=2 of them, which
    the check above already catches directly. A surname-core propagation
    step was tried and intentionally dropped for this reason: it cannot
    change any output here and would be complexity with no effect."""
    weak_keys = {key for key in key_to_indices if _is_weak_key(key)}
    ambiguous: set[str] = set()
    for key in weak_keys:
        key_tokens = key.split()
        tags = {tag for index in key_to_indices[key] if (tag := _best_tag_for_key(mention_keys, index, key, key_tokens))}
        relationship_tags = {tag for tag in tags if not tag.startswith("other:")}
        distinguishing_tags = {tag for tag in tags if tag.startswith("other:")}
        if len(relationship_tags) >= 2 or len(distinguishing_tags) >= 2 or (relationship_tags and distinguishing_tags):
            ambiguous.add(key)
    return ambiguous


def _resolve_ambiguous_keys(
    ambiguous_keys: set[str], key_to_indices: dict[str, list[int]], mention_keys: list[dict[str, list[str]]], parent: list[int]
) -> None:
    """For each ambiguous (weak, multi-identity) key: mentions sharing the
    SAME tag always unify with each other -- agreeing repeatedly that "M.
    Danglars" means the same thing is not a conflict, it's the normal case,
    and requiring pairwise corroboration even within one consistent tag-
    group would needlessly fragment it into singletons. Bare (untagged)
    mentions join the corpus's most frequent ("anchor") tag-group freely (a
    bare reference carries no information contradicting the default
    reading). Only a DIFFERENT, minority tag-group unifies with the anchor
    -- or with another minority tag-group -- when at least one member of
    each side shares a second key with a member of the other side, checked
    against those SPECIFIC mentions' own keysets, never an aggregated
    component (which would accumulate enough unrelated keys to spuriously
    "corroborate" almost any pairing)."""
    for key in ambiguous_keys:
        key_tokens = key.split()
        indices = key_to_indices[key]
        tag_of = {index: _best_tag_for_key(mention_keys, index, key, key_tokens) for index in indices}

        tag_groups: dict[str | None, list[int]] = defaultdict(list)
        for index in indices:
            tag_groups[tag_of[index]].append(index)

        tag_counts = {tag: len(members) for tag, members in tag_groups.items() if tag is not None}
        anchor_tag = max(sorted(tag_counts), key=lambda tag: tag_counts[tag]) if tag_counts else None

        # Bare mentions (tag None) are folded into the anchor group -- they
        # carry no conflicting signal, so they're never a separate group.
        anchor_members = tag_groups.get(anchor_tag, []) + tag_groups.get(None, [])
        for member in anchor_members[1:]:
            _union(parent, anchor_members[0], member)

        minority_groups = [members for tag, members in tag_groups.items() if tag is not None and tag != anchor_tag]
        for group in minority_groups:
            # Always unify members who agree on the same minority tag --
            # that is consistency, not ambiguity.
            for member in group[1:]:
                _union(parent, group[0], member)
            # Corroborate this minority group against the anchor group, and
            # against every other minority group, before joining them.
            for other_group in [anchor_members, *minority_groups]:
                if other_group is group:
                    continue
                if any((mention_keys[i].keys() & mention_keys[j].keys()) - {key} for i in group for j in other_group):
                    _union(parent, group[0], other_group[0])


def compute_components(entity_facts: list[ResolvedFact]) -> list[list[ResolvedFact]]:
    """Group entity_facts into connected components by name/alias equality,
    refined by Clause A (corroborated-alias rule) and Clause B
    (weak-surname rule) -- see module docstring.

    Returns components in a deterministic order (sorted by the minimum
    entity_id their members will eventually produce is not yet known at
    this stage, so components are returned in an arbitrary-but-deterministic
    order here; resolve_entities sorts the final EntityRecords by entity_id,
    which is the order that actually matters to callers).
    """
    if not entity_facts:
        return []

    ordered = sorted(entity_facts, key=_mention_sort_key)
    n = len(ordered)
    parent = list(range(n))

    # Clause A: an alias contributes a merge key only if its normalized
    # form ALSO appears, elsewhere in the corpus, as some mention's
    # normalized PRIMARY name. A mention's own primary name is always a
    # valid key -- Clause A is specifically about untrusted alias claims,
    # never about what a mention asserts of itself.
    primary_keys_corpus: set[str] = set()
    for fact in ordered:
        primary_key = norm_name(fact.raw.name)
        if primary_key and not is_generic_referent(primary_key):
            primary_keys_corpus.add(primary_key)

    # mention_keys[i]: key -> every raw surface string that produced it for
    # this mention (primary name first, then aliases in encounter order).
    mention_keys: list[dict[str, list[str]]] = []
    for fact in ordered:
        raw = fact.raw
        keys: dict[str, list[str]] = {}
        primary_key = norm_name(raw.name)
        if primary_key and not is_generic_referent(primary_key):
            keys.setdefault(primary_key, []).append(raw.name)
        for alias in raw.aliases:
            alias_key = norm_name(alias)
            if alias_key and not is_generic_referent(alias_key) and alias_key in primary_keys_corpus:
                keys.setdefault(alias_key, []).append(alias)
        mention_keys.append(keys)

    key_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, keys in enumerate(mention_keys):
        for key in keys:
            key_to_indices[key].append(index)

    ambiguous_keys = _detect_ambiguous_keys(key_to_indices, mention_keys)

    for key, indices in key_to_indices.items():
        if key in ambiguous_keys:
            continue
        first = indices[0]
        for other in indices[1:]:
            _union(parent, first, other)

    _resolve_ambiguous_keys(ambiguous_keys, key_to_indices, mention_keys, parent)

    components_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(n):
        components_by_root[_find(parent, index)].append(index)

    return [[ordered[i] for i in indices] for indices in components_by_root.values()]


def select_canonical_name(members: list[ResolvedFact]) -> str:
    """Tiebreak chain: most frequent surface form (each member's raw.name,
    not its aliases) -> smallest first-appearance char offset -> lexicographic.
    """
    name_counts: dict[str, int] = defaultdict(int)
    name_earliest_offset: dict[str, int] = {}

    for member in members:
        name = member.raw.name
        name_counts[name] += 1
        earliest = min(p.char_start for p in member.provenance)
        if name not in name_earliest_offset or earliest < name_earliest_offset[name]:
            name_earliest_offset[name] = earliest

    return min(name_counts, key=lambda name: (-name_counts[name], name_earliest_offset[name], name))


def select_entity_type(members: list[ResolvedFact]) -> EntityType:
    """Majority entity_type across the component; a tie resolves to OTHER."""
    counts: dict[EntityType, int] = defaultdict(int)
    for member in members:
        counts[member.raw.type] += 1

    best_count = max(counts.values())
    winners = [entity_type for entity_type, count in counts.items() if count == best_count]
    return winners[0] if len(winners) == 1 else EntityType.OTHER


def merge_component(members: list[ResolvedFact]) -> EntityRecord:
    """Build the final EntityRecord for one connected component."""
    canonical_name = select_canonical_name(members)
    entity_type = select_entity_type(members)
    entity_id = make_entity_id(canonical_name, entity_type.value)

    provenance = tuple(
        sorted({provenance for member in members for provenance in member.provenance}, key=lambda p: (p.chunk_id, p.char_start))
    )

    return EntityRecord(entity_id=entity_id, canonical_name=canonical_name, entity_type=entity_type, provenance=provenance)
