"""Connected-components merge, canonical-name tiebreak, and entity-type
majority-vote selection."""

from lncvs.graph.entity_resolution.merge import compute_components, merge_component, select_canonical_name, select_entity_type
from lncvs.graph.provenance.matching import MatchTier, QuoteMatch
from lncvs.graph.provenance.models import ResolvedFact
from lncvs.schemas import EntityType, Provenance


def _mention(
    local_id: str,
    name: str,
    chapter_index: int,
    window_index: int | None,
    chunk_id: str,
    char_start: int,
    aliases: tuple[str, ...] = (),
    entity_type: EntityType = EntityType.PERSON,
) -> ResolvedFact:
    from lncvs.graph.llm_extraction.schema import RawEntityMention

    raw = RawEntityMention(local_id=local_id, name=name, type=entity_type, aliases=aliases, evidence_quotes=(name,))
    provenance = (Provenance(chunk_id=chunk_id, char_start=char_start, char_end=char_start + len(name)),)
    quote_match = QuoteMatch(quote=name, tier=MatchTier.EXACT, char_start=0, char_end=len(name))
    return ResolvedFact(raw=raw, chapter_index=chapter_index, window_index=window_index, provenance=provenance, quote_matches=(quote_match,))


def test_identical_names_across_windows_merge_into_one_component() -> None:
    mentions = [
        _mention("e1", "John", 0, None, "c1", 0),
        _mention("e1", "John", 1, None, "c2", 100),
    ]
    components = compute_components(mentions)
    assert len(components) == 1
    assert len(components[0]) == 2


def test_distinct_names_do_not_merge() -> None:
    mentions = [
        _mention("e1", "John", 0, None, "c1", 0),
        _mention("e1", "Mary", 1, None, "c2", 100),
    ]
    components = compute_components(mentions)
    assert len(components) == 2


def test_alias_in_one_window_links_to_canonical_name_in_another() -> None:
    mentions = [
        _mention("e1", "Edmond Dantès", 0, None, "c1", 0, aliases=("Sinbad",)),
        _mention("e1", "Sinbad", 5, None, "c2", 200),
    ]
    components = compute_components(mentions)
    assert len(components) == 1
    assert len(components[0]) == 2


def test_transitive_merge_via_shared_alias_chain() -> None:
    """A <-> B via a shared alias, B <-> C via a different shared alias --
    all three must end up in one component (transitive closure). Uses
    proper-name aliases ("Sinbad") rather than a generic role noun, since
    generic referents are deliberately excluded from bridging (see
    test_generic_referent_aliases_do_not_bridge_distinct_entities)."""
    mentions = [
        _mention("e1", "Edmond Dantès", 0, None, "c1", 0, aliases=("Sinbad",)),
        _mention("e2", "Sinbad", 1, None, "c2", 50, aliases=("Monte Cristo",)),
        _mention("e3", "Monte Cristo", 2, None, "c3", 100),
    ]
    components = compute_components(mentions)
    assert len(components) == 1
    assert len(components[0]) == 3


def test_titles_do_not_prevent_or_force_a_merge_incorrectly() -> None:
    mentions = [
        _mention("e1", "Lord Glenarvan", 0, None, "c1", 0),
        _mention("e1", "Glenarvan", 1, None, "c2", 100),
        _mention("e1", "Lady Helena", 2, None, "c3", 200),
    ]
    components = compute_components(mentions)
    # Glenarvan-with-title and Glenarvan-without-title merge; Helena is distinct.
    sizes = sorted(len(c) for c in components)
    assert sizes == [1, 2]


def test_components_partition_all_mentions_with_no_loss() -> None:
    mentions = [
        _mention("e1", "John", 0, None, "c1", 0),
        _mention("e1", "Mary", 0, None, "c1", 10),
        _mention("e1", "John", 1, None, "c2", 0),
    ]
    components = compute_components(mentions)
    total = sum(len(c) for c in components)
    assert total == len(mentions)


# --- generic-referent guard: the catastrophic-over-merge regression fix ---


def test_generic_referent_aliases_do_not_bridge_distinct_entities() -> None:
    """The dominant real-corpus failure mode: two unrelated characters
    both carry a generic pronoun ("he") as an alias. They must NOT merge
    through it -- otherwise a single shared "he"/"him"/"the major" key
    transitively collapses an entire novel's cast into one node."""
    mentions = [
        _mention("e1", "Paganel", 0, None, "c1", 0, aliases=("he", "him")),
        _mention("e2", "Glenarvan", 1, None, "c2", 100, aliases=("he", "him")),
        _mention("e3", "Ayrton", 2, None, "c3", 200, aliases=("the major", "this man")),
    ]
    components = compute_components(mentions)
    assert len(components) == 3


def test_kinship_and_role_referents_do_not_bridge() -> None:
    """Kinship terms ("my father") and bare role nouns ("chief") are
    non-identifying and must never bridge two distinct named entities."""
    mentions = [
        _mention("e1", "Villefort", 0, None, "c1", 0, aliases=("my father", "the magistrate")),
        _mention("e2", "Morrel", 1, None, "c2", 100, aliases=("my father", "the merchant")),
        _mention("e3", "Kai-Koumou", 2, None, "c3", 200, aliases=("chief", "the native")),
        _mention("e4", "Kara-Tété", 3, None, "c4", 300, aliases=("chief",)),
    ]
    components = compute_components(mentions)
    assert len(components) == 4


def test_real_alias_still_bridges_even_when_a_generic_alias_is_also_present() -> None:
    """The generic guard must be surgical: a genuine shared proper-name
    alias still merges, even if both mentions ALSO list a generic alias."""
    mentions = [
        _mention("e1", "Ayrton", 0, None, "c1", 0, aliases=("Ben Joyce", "he")),
        _mention("e2", "Ben Joyce", 1, None, "c2", 100, aliases=("him",)),
    ]
    components = compute_components(mentions)
    assert len(components) == 1
    assert len(components[0]) == 2


def test_titled_compound_name_merges_with_bare_surname() -> None:
    """Expanded title stripping: "Major MacNabb" and "King Louis XVI."
    normalize to their bare names and merge with the untitled mention,
    while a bare title alone ("the major") never bridges."""
    macnabb = compute_components(
        [
            _mention("e1", "Major MacNabb", 0, None, "c1", 0),
            _mention("e1", "MacNabb", 1, None, "c2", 100),
        ]
    )
    assert len(macnabb) == 1 and len(macnabb[0]) == 2

    louis = compute_components(
        [
            _mention("e1", "King Louis XVI.", 0, None, "c1", 0),
            _mention("e1", "Louis XVI", 1, None, "c2", 100),
        ]
    )
    assert len(louis) == 1 and len(louis[0]) == 2


def test_mention_named_only_by_a_generic_referent_becomes_a_singleton() -> None:
    """A mention whose every name/alias is generic contributes no merge
    key and stands alone -- benign noise, never a wrong-character bridge."""
    mentions = [
        _mention("e1", "the major", 0, None, "c1", 0, aliases=("he",)),
        _mention("e2", "this man", 1, None, "c2", 100, aliases=("him",)),
    ]
    components = compute_components(mentions)
    assert len(components) == 2


def test_empty_input_returns_no_components() -> None:
    assert compute_components([]) == []


def test_select_canonical_name_prefers_most_frequent_surface_form() -> None:
    members = [
        _mention("e1", "John", 0, None, "c1", 0),
        _mention("e1", "John", 1, None, "c2", 10),
        _mention("e1", "Johnny", 2, None, "c3", 20),
    ]
    assert select_canonical_name(members) == "John"


def test_select_canonical_name_breaks_frequency_tie_by_earliest_offset() -> None:
    members = [
        _mention("e1", "Johnny", 0, None, "c1", 500),
        _mention("e1", "John", 1, None, "c2", 10),  # earlier offset, same frequency (1 each)
    ]
    assert select_canonical_name(members) == "John"


def test_select_canonical_name_breaks_remaining_tie_lexicographically() -> None:
    members = [
        _mention("e1", "Zed", 0, None, "c1", 5),
        _mention("e1", "Amy", 1, None, "c2", 5),  # identical offset and frequency
    ]
    assert select_canonical_name(members) == "Amy"


def test_select_entity_type_majority_vote() -> None:
    members = [
        _mention("e1", "John", 0, None, "c1", 0, entity_type=EntityType.PERSON),
        _mention("e1", "John", 1, None, "c2", 10, entity_type=EntityType.PERSON),
        _mention("e1", "John", 2, None, "c3", 20, entity_type=EntityType.OBJECT),
    ]
    assert select_entity_type(members) is EntityType.PERSON


def test_select_entity_type_tie_resolves_to_other() -> None:
    members = [
        _mention("e1", "Duncan", 0, None, "c1", 0, entity_type=EntityType.OBJECT),
        _mention("e1", "Duncan", 1, None, "c2", 10, entity_type=EntityType.LOCATION),
    ]
    assert select_entity_type(members) is EntityType.OTHER


def test_merge_component_produces_deterministic_entity_id() -> None:
    members = [_mention("e1", "John", 0, None, "c1", 0)]
    first = merge_component(members)
    second = merge_component(members)
    assert first.entity_id == second.entity_id
    assert first == second


def test_merge_component_unions_provenance_across_members() -> None:
    members = [
        _mention("e1", "John", 0, None, "c1", 0),
        _mention("e1", "John", 1, None, "c2", 100),
    ]
    entity = merge_component(members)
    assert {p.chunk_id for p in entity.provenance} == {"c1", "c2"}


# --- ER4 Clause A: corroborated-alias rule ---


def test_epithet_alias_never_merges_two_characters() -> None:
    """An alias that is never anyone's PRIMARY name (a descriptive
    epithet the LLM invented, e.g. "the dying man") must not bridge two
    different characters, even though it isn't a generic referent and
    would have passed the pre-ER4 filter."""
    mentions = [
        _mention("e1", "Faria", 0, None, "c1", 0, aliases=("the dying man",)),
        _mention("e2", "Caderousse", 1, None, "c2", 100, aliases=("the dying man",)),
    ]
    components = compute_components(mentions)
    assert len(components) == 2


def test_genuine_alias_still_merges_because_it_is_corroborated() -> None:
    """A real identity alias ("Monte Cristo") DOES merge -- it is
    corroborated because some mention's own PRIMARY name is exactly
    that string, unlike an invented epithet."""
    mentions = [
        _mention("e1", "Dantès", 0, None, "c1", 0, aliases=("Monte Cristo",)),
        _mention("e2", "Monte Cristo", 1, None, "c2", 100),
    ]
    components = compute_components(mentions)
    assert len(components) == 1
    assert len(components[0]) == 2


def test_a_mentions_own_primary_name_is_never_subject_to_corroboration() -> None:
    """Clause A constrains ALIASES only. A mention's own primary name is
    always a valid merge key, even if no OTHER mention ever uses that
    exact string -- requiring corroboration for primary names too would
    make the very first mention of any character unmergeable with itself
    on the next page."""
    mentions = [
        _mention("e1", "Paganel", 0, None, "c1", 0),
        _mention("e1", "Paganel", 1, None, "c2", 100),
    ]
    components = compute_components(mentions)
    assert len(components) == 1
    assert len(components[0]) == 2


def test_multiple_epithet_aliases_cannot_chain_three_characters_together() -> None:
    """Several different invented epithets, each shared by a different
    pair, must not transitively chain three unrelated characters into one
    component -- every individual epithet link must be rejected."""
    mentions = [
        _mention("e1", "Faria", 0, None, "c1", 0, aliases=("the dying man",)),
        _mention("e2", "Caderousse", 1, None, "c2", 100, aliases=("the dying man", "the bandit")),
        _mention("e3", "Vampa", 2, None, "c3", 200, aliases=("the bandit",)),
    ]
    components = compute_components(mentions)
    assert len(components) == 3


# --- ER4 Clause B: weak-surname rule ---


def test_bare_surname_alone_does_not_merge_two_distinguishable_family_members() -> None:
    """Two different family members, each carrying a DIFFERENT
    distinguishing qualifier on the same bare surname (a gender-coded
    honorific contrast: Baron vs Mademoiselle), must not merge merely
    because both are sometimes called by the bare surname alone."""
    mentions = [
        _mention("e1", "Baron Danglars", 0, None, "c1", 0, aliases=("Danglars",)),
        _mention("e2", "Mademoiselle Danglars", 1, None, "c2", 100, aliases=("Danglars",)),
    ]
    components = compute_components(mentions)
    assert len(components) == 2


def test_surname_plus_corroborating_identifier_still_merges() -> None:
    """Once a SECOND shared key corroborates the connection (here, both
    mentions also share the alias "Hercule", corroborated as a valid
    alias key by e3's matching primary name), the surname-only ambiguity
    is resolved and they merge -- corroboration, not an absolute ban on
    surname-based merging."""
    mentions = [
        _mention("e1", "Baron Danglars", 0, None, "c1", 0, aliases=("Danglars", "Hercule")),
        _mention("e2", "Mademoiselle Danglars", 1, None, "c2", 100, aliases=("Danglars", "Hercule")),
        _mention("e3", "Hercule", 2, None, "c3", 200),
    ]
    components = compute_components(mentions)
    assert len(components) == 1
    assert len(components[0]) == 3


def test_full_name_merge_is_unaffected_by_the_weak_surname_rule() -> None:
    """A full, non-particle-led, multi-token name ("Edmond Dantès") is
    STRONG evidence under Clause B and merges unconditionally, exactly as
    before -- the weak-surname rule applies only to bare/particle-led
    surname keys."""
    mentions = [
        _mention("e1", "Edmond Dantès", 0, None, "c1", 0),
        _mention("e2", "Edmond Dantès", 1, None, "c2", 100),
    ]
    components = compute_components(mentions)
    assert len(components) == 1
    assert len(components[0]) == 2


def test_same_as_style_identity_aliases_are_preserved_through_clause_b() -> None:
    """A character known under several full, non-particle-led aliases
    (the SAME_AS-style identity-reveal pattern -- Ayrton is Ben Joyce)
    still merges into one component; Clause B's weak-surname handling
    must never interfere with this, since neither "Ayrton" nor "Ben
    Joyce" is a bare-surname/particle-led key."""
    mentions = [
        _mention("e1", "Ayrton", 0, None, "c1", 0, aliases=("Ben Joyce",)),
        _mention("e2", "Ben Joyce", 1, None, "c2", 100),
    ]
    components = compute_components(mentions)
    assert len(components) == 1
    assert len(components[0]) == 2


def test_a_surname_used_consistently_by_one_person_is_not_flagged_ambiguous() -> None:
    """A surname always paired with the SAME qualifier (never a second,
    conflicting one) is not ambiguous and merges like any other key --
    most characters are simply always called by one bare/titled name, and
    that is the normal case, not a family collision."""
    mentions = [
        _mention("e1", "Major MacNabb", 0, None, "c1", 0, aliases=("MacNabb",)),
        _mention("e2", "MacNabb", 1, None, "c2", 100),
        _mention("e3", "Major MacNabb", 2, None, "c3", 200),
    ]
    components = compute_components(mentions)
    assert len(components) == 1
    assert len(components[0]) == 3


def test_bare_mentions_join_the_majority_qualifier_without_needing_corroboration() -> None:
    """A BARE mention (no distinguishing qualifier at all) carries no
    information that contradicts the dominant reading of an ambiguous
    surname, so it joins the majority ("anchor") group freely -- only a
    genuinely conflicting MINORITY qualifier needs corroboration."""
    mentions = [
        _mention("e1", "Baron Danglars", 0, None, "c1", 0),
        _mention("e2", "Baron Danglars", 1, None, "c2", 100),
        _mention("e3", "Baron Danglars", 2, None, "c3", 200),
        _mention("e4", "Danglars", 3, None, "c4", 300),  # bare -- should join the Baron majority
        _mention("e5", "Mademoiselle Danglars", 4, None, "c5", 400),  # minority, conflicting
    ]
    components = compute_components(mentions)
    sizes = sorted(len(c) for c in components)
    # The 3 Baron mentions + the bare mention join one component (4);
    # Mademoiselle Danglars, uncorroborated, stays separate (1).
    assert sizes == [1, 4]


def test_mentions_sharing_the_same_minority_qualifier_merge_with_each_other() -> None:
    """Multiple mentions that all agree on the SAME minority qualifier are
    not ambiguous with respect to EACH OTHER -- only the cross-qualifier
    boundary (minority vs anchor) needs corroboration. Requiring pairwise
    corroboration even within one consistent minority group would
    needlessly fragment it into singletons."""
    mentions = [
        _mention("e1", "Baron Danglars", 0, None, "c1", 0),  # anchor (majority)
        _mention("e2", "Baron Danglars", 1, None, "c2", 100),
        _mention("e3", "Mademoiselle Danglars", 2, None, "c3", 200),  # minority
        _mention("e4", "Mademoiselle Danglars", 3, None, "c4", 300),  # same minority
    ]
    components = compute_components(mentions)
    sizes = sorted(len(c) for c in components)
    assert sizes == [2, 2]


def test_gender_neutral_title_does_not_create_a_false_split() -> None:
    """A gender-neutral-ish formal title ("M.", "Monsieur") does not
    distinguish WHICH family member is meant, so it must not be treated
    as its own exclusive qualifier category -- "M. Danglars" and "Baron
    Danglars" are the same man, not a conflict."""
    mentions = [
        _mention("e1", "Baron Danglars", 0, None, "c1", 0),
        _mention("e2", "M. Danglars", 1, None, "c2", 100),
    ]
    components = compute_components(mentions)
    assert len(components) == 1
    assert len(components[0]) == 2


def test_uncorroborated_bare_alias_of_a_titled_surname_does_not_merge() -> None:
    """An alias that reduces to a bare surname ("Morcerf") is itself
    subject to Clause A: it only counts as a merge key if "Morcerf" is
    ALSO some mention's own primary name somewhere in the corpus. Neither
    mention here has that exact primary name, so the alias is dropped
    entirely and they do not merge through it -- a Clause A effect, not
    Clause B's ambiguity handling."""
    mentions = [
        _mention("e1", "Albert de Morcerf", 0, None, "c1", 0, aliases=("Morcerf",)),
        _mention("e2", "Countess of Morcerf", 1, None, "c2", 100, aliases=("Morcerf",)),
    ]
    components = compute_components(mentions)
    assert len(components) == 2
