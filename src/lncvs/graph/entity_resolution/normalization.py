"""Deterministic name normalization for cross-window entity matching
(frozen G2 spec §4).

norm_name() is the sole basis for deciding two entity mentions refer to
the same entity -- no fuzzy similarity, no embedding distance. Two names
match iff their normalized forms are exactly equal.

is_generic_referent() is the companion guard that merge.py applies to
every normalized merge key. It exists to fix the dominant entity-resolution
failure mode observed on the real hackathon corpus: the LLM extractor
routinely emits generic referents -- bare pronouns ("he", "him"),
demonstratives ("this man"), bare ranks/titles ("the major", "chief"),
and kinship terms ("my father", "his son") -- as a mention's name or in
its alias list. Because the frozen merge policy unions any two mentions
that share a normalized key, these non-identifying tokens act as universal
bridges, transitively collapsing the entire principal cast of a novel into
one giant entity (e.g. Paganel, Glenarvan, Ayrton AND Ben Joyce merged
into a single node). That is precisely the catastrophic false-merge the
frozen spec §4 was written to avoid ("This deliberately biases toward
false splits over false merges -- ... a false merge manufactures the
wrong-character-evidence failure mode"). Excluding purely-generic keys
serves that stated intent: it only ever *prevents* a merge (a mention
referred to solely by a generic term becomes its own component, a benign
false split that merely reduces multi-hop reach), and never forces one.
The policy itself -- name/alias string equality, no fuzzy/embedding
matching -- is unchanged.
"""

# Rank / noble / honorific titles stripped from the FRONT of a multi-token
# name. Two distinct, deliberate effects: (1) a titled compound like
# "Major MacNabb" or "King Louis XVI." normalizes to the bare name
# ("macnabb" / "louis xvi") so it correctly merges with the untitled form;
# (2) a bare title used alone ("the major") reduces to "" and is dropped as
# a merge key. Every token here is unambiguously a title, never a standalone
# personal identity, so stripping it can only prevent a spurious bridge,
# never erase a real name. Kinship terms (father/son/...) are deliberately
# NOT here -- they are handled by is_generic_referent instead, because
# "father" doubles as the relational referent "my father" and must not be
# stripped as if it prefixed a real name.
_TITLE_PREFIXES = frozenset(
    {
        # original frozen set
        "mr", "mrs", "miss", "lord", "lady", "captain", "count", "countess",
        "abbe", "abbé", "m", "mlle", "mme", "dr", "the",
        # rank / military
        "major", "general", "colonel", "lieutenant", "sergeant", "admiral",
        "commander", "commodore", "corporal",
        # nobility / honorific
        "sir", "king", "queen", "prince", "princess", "duke", "duchess",
        "baron", "baroness", "marquis", "marquise", "viscount", "vicomte",
        "earl", "monsieur", "madame", "mademoiselle", "don", "doña", "dona",
        "señor", "senor", "señora", "senora", "herr", "frau", "signor",
        # clerical / academic
        "abbot", "reverend", "rev", "saint", "st", "professor", "prof",
    }
)

# Non-identifying generic referents. A merge key composed *entirely* of
# these tokens carries no entity identity and must never bridge two
# mentions. Pronouns, demonstratives, determiners, generic person/role
# nouns, and kinship terms -- the empirically-observed bridge tokens on the
# real corpus. Deliberately conservative: tokens that could plausibly be a
# real surname are excluded, and any title already in _TITLE_PREFIXES is
# omitted here because a bare title already normalizes to "" and is dropped
# anyway.
_GENERIC_REFERENT_TOKENS = frozenset(
    {
        # pronouns / reflexives
        "he", "him", "his", "she", "her", "hers", "it", "its",
        "they", "them", "their", "theirs",
        "i", "me", "my", "mine", "you", "your", "yours", "we", "us", "our", "ours",
        "himself", "herself", "itself", "themselves", "myself", "yourself", "ourselves",
        # demonstratives / determiners / quantifiers
        "this", "that", "these", "those", "who", "whom", "whose", "which", "what",
        "a", "an", "the", "latter", "former", "other", "others", "another", "same", "such",
        "one", "ones", "someone", "somebody", "anyone", "anybody",
        "everyone", "everybody", "none", "nobody", "all", "both", "each", "every",
        # adjectival modifiers that only ever qualify a generic noun in these
        # referents ("young man", "old man", "poor fellow") -- a key is dropped
        # only if ALL its tokens are generic, so "young Dantès"/"little Edward"
        # (a real name token present) is unaffected
        "young", "old", "little", "poor", "good", "great", "dear", "new",
        "certain", "late", "honest", "brave", "worthy", "noble", "first", "second",
        # honorific abstract referents ("your excellency", "his lordship",
        # "her majesty") -- the possessive is already generic above
        "excellency", "excellence", "lordship", "ladyship", "majesty", "highness",
        "grace", "honour", "honor", "worship", "eminence", "reverence", "self",
        # generic person nouns
        "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
        "child", "children", "person", "persons", "people", "individual",
        "gentleman", "gentlemen", "fellow", "stranger", "strangers",
        "guest", "guests", "friend", "friends", "companion", "companions",
        # generic roles / collectives
        "chief", "officer", "officers", "soldier", "soldiers", "guard", "guards",
        "sailor", "sailors", "servant", "servants", "steward", "colonist", "colonists",
        "peasant", "peasants", "native", "natives", "crew", "party", "troop", "troops",
        "prisoner", "prisoners", "quartermaster", "doctor", "priest", "host",
        "master", "mistress", "band", "group", "settlers", "squatters", "warriors",
        "workman", "workmen",
        # generic occupational / descriptive role nouns observed bridging
        # distinct named characters in the real corpus (each is a common
        # noun used by the extractor as a name/alias, never a proper name)
        "geographer", "leader", "traveller", "traveler", "magistrate", "deputy",
        "attorney", "banker", "merchant", "assassin", "murderer", "thief", "robber",
        "smuggler", "jailer", "turnkey", "gendarme", "notary", "clerk", "governor",
        "emperor", "usurper", "tyrant", "foreigner", "neighbour", "neighbor", "unknown",
        # demonyms / nationalities used as standalone referents
        "frenchman", "frenchmen", "englishman", "englishmen", "scotchman", "scotsman",
        "irishman", "italian", "italians", "spaniard", "corsican", "catalan", "catalans",
        "greek", "maltese", "roman", "parisian", "european", "europeans", "australians",
        # generic object / place class nouns (keep distinct ships, houses, rooms
        # from collapsing into one node; multi-token proper names like
        # "house of morcerf" survive because not every token is generic)
        "yacht", "vessel", "ship", "ships", "boat", "house", "chamber", "room",
        "apartment", "letter", "island", "carriage",
        # kinship
        "father", "mother", "son", "sons", "daughter", "daughters",
        "brother", "brothers", "sister", "sisters", "wife", "husband",
        "grandfather", "grandmother", "grandson", "granddaughter",
        "grandchild", "grandchildren", "uncle", "aunt", "cousin",
        "parent", "parents", "niece", "nephew", "spouse",
    }
)


def norm_name(name: str) -> str:
    """Lowercase, strip punctuation (replaced with spaces), collapse
    whitespace, then strip any number of leading title-prefix tokens."""
    lowered = name.lower().strip()
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in lowered)
    tokens = cleaned.split()

    while tokens and tokens[0] in _TITLE_PREFIXES:
        tokens.pop(0)

    return " ".join(tokens)


def is_generic_referent(normalized_name: str) -> bool:
    """True if normalized_name carries no entity identity -- i.e. it is
    empty or every one of its tokens is a generic referent (pronoun,
    determiner, generic person/role noun, or kinship term).

    Applied by merge.py to every candidate merge key: a key for which this
    returns True is never used to union mentions, so two distinct
    characters can never be merged solely because both were referred to as
    "he", "the major", or "my father". A name with at least one specific
    token ("major macnabb" -> "macnabb"; "Louis XVI" -> "louis xvi") is
    NOT generic and is kept as a normal merge key.
    """
    tokens = normalized_name.split()
    if not tokens:
        return True
    return all(token in _GENERIC_REFERENT_TOKENS for token in tokens)
