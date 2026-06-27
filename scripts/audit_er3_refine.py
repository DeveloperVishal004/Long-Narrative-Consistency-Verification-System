"""ER3 refinement (READ-ONLY): separate epithet-alias bridges from genuine
surname collision, and counterfactually measure which deterministic rule
shrinks the giant component most. No code changes; analysis only."""

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_er3_merge_keys import GIVEN_NAMES, NOBILIARY_PARTICLES, build_components, load_mentions  # noqa: E402


def giant(mentions, exclude):
    comps, _, _ = build_components(mentions, exclude_keys=exclude)
    return max(len(c) for c in comps), len(comps)


def run(book, cf, sid):
    mentions = load_mentions(book, cf, sid)
    base, nbase = giant(mentions, frozenset())

    primary_keys = {m.name_key for m in mentions if m.name_key}
    all_keys = set()
    for m in mentions:
        all_keys.update(m.keys)

    # classify every key
    alias_only = {k for k in all_keys if k not in primary_keys}          # never a primary name
    corroborated = {k for k in all_keys if k in primary_keys}            # is some mention's primary name
    single_tok = {k for k in all_keys if len(k.split()) == 1}
    particle = {k for k in all_keys if len(k.split()) > 1 and k.split()[0] in NOBILIARY_PARTICLES}
    given_only = {k for k in single_tok if k in GIVEN_NAMES}
    surname_like = (single_tok - given_only) | particle                 # bare surname or particle+surname
    surname_corroborated = surname_like & corroborated                  # surnames that are real primary names

    # counterfactual rules (each = which keys we FORBID from bridging)
    rules = {
        "BASELINE (current)": frozenset(),
        "R1: ignore ALL alias-only keys (epithets + alias-only names)": frozenset(alias_only),
        "R2: ignore bare-surname / particle keys": frozenset(surname_like),
        "R3: R1 + R2 (alias-only AND surname-like forbidden)": frozenset(alias_only | surname_like),
        "R4: merge ONLY on multi-token full-name keys": frozenset(k for k in all_keys if len(k.split()) == 1 or k.split()[0] in NOBILIARY_PARTICLES),
    }
    print("=" * 80)
    print(f"{book}")
    print("=" * 80)
    print(f"distinct merge keys: {len(all_keys)}  | alias-only(epithet-ish): {len(alias_only)}  "
          f"| corroborated(also-primary): {len(corroborated)}")
    print(f"surname-like keys: {len(surname_like)} (of which also a real primary name: {len(surname_corroborated)})")
    print()
    print(f"{'rule':62s} {'giant':>6s} {'#comp':>6s}  Δgiant")
    for name, excl in rules.items():
        g, nc = giant(mentions, excl)
        print(f"{name:62s} {g:6d} {nc:6d}  {base - g:+d}")
    print()

    # Per character-edge in the giant: is it bridged ONLY by epithet (alias-only)
    # keys, ONLY by surname-like keys, or does it have a strong (full-name or
    # corroborated-name) link? This partitions the *incorrect* merges.
    comps, _, _ = build_components(mentions, frozenset())
    g = max(comps, key=len)
    name_keys = defaultdict(set)
    for m in g:
        for k in m.keys:
            name_keys[m.primary].add(k)
    key_names = defaultdict(set)
    for nm, ks in name_keys.items():
        for k in ks:
            key_names[k].add(nm)
    edge_keys = defaultdict(set)
    for k, nms in key_names.items():
        nl = sorted(nms)
        for i in range(len(nl)):
            for j in range(i + 1, len(nl)):
                edge_keys[(nl[i], nl[j])].add(k)

    def edge_class(ks):
        has_full = any(len(k.split()) > 1 and k.split()[0] not in NOBILIARY_PARTICLES and k in corroborated for k in ks)
        has_corrob_strong = any(k in corroborated and (len(k.split()) > 1) for k in ks)
        only_alias = all(k in alias_only for k in ks)
        only_surname = all(k in surname_like for k in ks)
        if has_full or has_corrob_strong:
            return "strong (full/corroborated name) -- likely legit"
        if only_alias:
            return "ONLY alias-only/epithet keys"
        if only_surname:
            return "ONLY surname-like keys"
        return "mixed weak (surname+given+epithet, no full name)"

    cls_counts = Counter(edge_class(ks) for ks in edge_keys.values())
    tot = sum(cls_counts.values())
    print(f"Character-pair bridges in giant ({tot} pairs) by strongest evidence:")
    for c, n in cls_counts.most_common():
        print(f"   {c:55s} {n:5d}  ({100*n/tot:.1f}%)")

    # concrete epithet keys driving bridges (alias-only, multi-token or non-name)
    epithet_bridge = []
    for k in alias_only:
        nms = key_names.get(k, set())
        if len(nms) >= 2:
            epithet_bridge.append((k, len(nms), sorted(nms)[:5]))
    epithet_bridge.sort(key=lambda x: -x[1])
    print(f"\nTop alias-only (epithet) keys bridging >=2 distinct characters in giant:")
    for k, n, s in epithet_bridge[:20]:
        print(f"   {k!r:32s} bridges {n} chars -> {s}")

    # surname keys bridging >=3 distinct family members
    surname_bridge = []
    for k in surname_like:
        nms = key_names.get(k, set())
        if len(nms) >= 3:
            surname_bridge.append((k, len(nms), sorted(nms)[:6]))
    surname_bridge.sort(key=lambda x: -x[1])
    print(f"\nTop surname-like keys bridging >=3 distinct characters in giant (family collision):")
    for k, n, s in surname_bridge[:15]:
        print(f"   {k!r:24s} bridges {n} chars -> {s}")
    print()


if __name__ == "__main__":
    run("In Search of the Castaways", "extraction_cache_in_search_of_the_castaways.jsonl", "castaways")
    run("The Count of Monte Cristo", "extraction_cache_the_count_of_monte_cristo.jsonl", "mc")
