"""ER4 prototype (READ-ONLY): test corroboration strictness levels against
the real cached corpus before touching merge.py. Validates against the
explicit preserve-list before any production change."""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_er3_merge_keys import NOBILIARY_PARTICLES, load_mentions  # noqa: E402


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
        if ra < rb:
            self.p[rb] = ra
        else:
            self.p[ra] = rb
        return True


def is_weak(key: str) -> bool:
    toks = key.split()
    if len(toks) == 1:
        return True
    if toks[0] in NOBILIARY_PARTICLES:
        return True
    return False


def corroborated_resolve(mentions, level):
    """level: 'pairwise' (mentions must share >=2 keys) or
    'component' (components must share >=2 distinct keys, fixpoint)."""
    n = len(mentions)
    primary_keys_corpus = {m.name_key for m in mentions if m.name_key}

    # Clause A: alias keys only valid if corroborated by appearing as a
    # primary name somewhere in the corpus.
    mention_keys = []
    for m in mentions:
        keys = set()
        if m.name_key:
            keys.add(m.name_key)
        for ak in m.alias_keys:
            if ak in primary_keys_corpus:
                keys.add(ak)
        mention_keys.append(keys)

    key_to_idx = defaultdict(list)
    for i, keys in enumerate(mention_keys):
        for k in keys:
            key_to_idx[k].append(i)

    strong_keys = {k for k in key_to_idx if not is_weak(k)}
    weak_keys = {k for k in key_to_idx if is_weak(k)}

    uf = UF(n)
    # Pass 1: strong keys always union.
    for k in strong_keys:
        idxs = key_to_idx[k]
        for o in idxs[1:]:
            uf.union(idxs[0], o)

    if level == "pairwise":
        # A weak key unions mention pair (a,b) only if they share >=2 keys total.
        for k in weak_keys:
            idxs = key_to_idx[k]
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a, b = idxs[i], idxs[j]
                    shared = mention_keys[a] & mention_keys[b]
                    if len(shared) >= 2:
                        uf.union(a, b)
    elif level == "component":
        # Fixpoint: a weak key unions two components iff those components
        # ALSO share some other key (any kind) elsewhere among their members.
        changed = True
        rounds = 0
        while changed and rounds < 10:
            changed = False
            rounds += 1
            # component -> set of keys held by any of its members
            comp_keys = defaultdict(set)
            comp_members = defaultdict(list)
            for i in range(n):
                r = uf.find(i)
                comp_keys[r].update(mention_keys[i])
                comp_members[r].append(i)
            for k in weak_keys:
                idxs = key_to_idx[k]
                roots = {uf.find(i) for i in idxs}
                if len(roots) < 2:
                    continue
                roots = sorted(roots)
                for i in range(len(roots)):
                    for j in range(i + 1, len(roots)):
                        ra, rb = roots[i], roots[j]
                        if uf.find(ra) == uf.find(rb):
                            continue
                        other_shared = (comp_keys[ra] & comp_keys[rb]) - {k}
                        if other_shared:
                            if uf.union(ra, rb):
                                changed = True
    else:
        raise ValueError(level)

    comps = defaultdict(list)
    for i in range(n):
        comps[uf.find(i)].append(mentions[i])
    return list(comps.values())


def check_preserve(comps, book):
    name_to_comp = {}
    for idx, c in enumerate(comps):
        for m in c:
            name_to_comp.setdefault(m.primary, set()).add(idx)
    print(f"  preserve-list check for {book}:")
    if "castaways" in book.lower():
        pairs = [("Ayrton", "Ben Joyce")]
    else:
        pairs = [
            ("Dantès", "Monte Cristo"), ("Dantès", "Edmond Dantès"),
            ("Lord Wilmore", "Dantès"), ("Abbé Busoni", "Dantès"),
            ("Sinbad the Sailor", "Dantès"),
        ]
    for a, b in pairs:
        ca = name_to_comp.get(a, set())
        cb = name_to_comp.get(b, set())
        ok = bool(ca & cb)
        print(f"    {a!r} <-> {b!r}: {'MERGED (preserved)' if ok else 'NOT MERGED (BROKEN!)'}  comps={ca or '?'}/{cb or '?'}")


def check_family_separation(comps, book):
    name_to_comp = {}
    for idx, c in enumerate(comps):
        for m in c:
            name_to_comp.setdefault(m.primary, set()).add(idx)
    print(f"  family-separation check for {book}:")
    if "castaways" in book.lower():
        groups = [("Captain Grant", "Mary Grant"), ("Captain Grant", "Harry Grant")]
    else:
        groups = [
            ("Baron Danglars", "Eugénie Danglars"), ("M. de Villefort", "Valentine"),
            ("Albert", "Count of Morcerf"), ("Andrea Cavalcanti", "Albert"),
        ]
    for a, b in groups:
        ca = name_to_comp.get(a, set())
        cb = name_to_comp.get(b, set())
        same = bool(ca & cb)
        print(f"    {a!r} vs {b!r}: {'STILL MERGED' if same else 'separated'}  comps={ca or '?'}/{cb or '?'}")


def run(book, cf, sid):
    mentions = load_mentions(book, cf, sid)
    print("=" * 78)
    print(book)
    print("=" * 78)
    for level in ["pairwise", "component"]:
        comps = corroborated_resolve(mentions, level)
        comps.sort(key=len, reverse=True)
        giant = comps[0]
        print(f"\n[{level}] components={len(comps)} giant_mentions={len(giant)} giant_distinct_names={len({m.primary for m in giant})}")
        check_preserve(comps, book)
        check_family_separation(comps, book)
    print()


if __name__ == "__main__":
    run("In Search of the Castaways", "extraction_cache_in_search_of_the_castaways.jsonl", "castaways")
    run("The Count of Monte Cristo", "extraction_cache_the_count_of_monte_cristo.jsonl", "mc")
