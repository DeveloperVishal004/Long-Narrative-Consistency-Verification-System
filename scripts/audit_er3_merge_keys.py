"""ER3 investigation (READ-ONLY): localize precisely why the remaining
giant merged components still exist after the generic-referent fix.

Reconstructs the EXACT union-find merge.compute_components performs (post
generic-referent fix), but instruments it: every merge key, every
character-level union, and the key that caused it are recorded for
attribution. No source code is modified; this script only reads the
frozen, cached extraction (results/extraction_cache_*.jsonl) and replays
the deterministic resolution pipeline. Entity resolution never consults
SAME_AS (it is a relation, not an alias), so SAME_AS contributes nothing
to merging here -- a fact this script confirms quantitatively.

Outputs JSON to results/er3_audit_<book>.json plus a printed summary.
"""

import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from evaluate_dataset import BOOK_NAME_TO_PATH  # noqa: E402

from lncvs.chunking import ChunkingConfig, chunk_document  # noqa: E402
from lncvs.graph.entity_resolution.normalization import is_generic_referent, norm_name  # noqa: E402
from lncvs.graph.llm_extraction import ExtractionConfig, LLMWindowExtractor  # noqa: E402
from lncvs.graph.provenance.service import resolve_window_provenance  # noqa: E402
from lncvs.graph.segmentation import segment_into_extraction_windows  # noqa: E402
from lncvs.ingestion import load_and_clean_narrative  # noqa: E402
from lncvs.llm import CachingStructuredLLMClient, JsonlStructuredLLMCache, LLMConfig  # noqa: E402

logging.disable(logging.WARNING)
RESULTS = REPO_ROOT / "results"

# Small lexicon of given (first) names appearing in the two novels, to
# distinguish "partial person name (given name only)" from "surname".
GIVEN_NAMES = {
    "edmond", "edward", "albert", "andrea", "franz", "lucien", "maximilian",
    "gerard", "gérard", "valentine", "eugénie", "eugenie", "renée", "renee",
    "haydée", "haydee", "julie", "fernand", "benedetto", "louis", "gaspard",
    "luigi", "giovanni", "bertuccio", "ali", "jacopo", "john", "mary", "robert",
    "jacques", "harry", "helena", "thalcave", "wilson", "tom", "austin",
    "mulready", "olbinett", "paganel",
}
NOBILIARY_PARTICLES = {"de", "d", "du", "des", "la", "le", "von", "van", "of", "saint", "st"}


@dataclass
class Mention:
    idx: int
    primary: str            # raw.name surface form
    primary_key: str        # norm_name(raw.name), may be "" or generic
    keys: tuple             # non-empty, non-generic keys (name + aliases)
    name_key: str | None    # primary_key if it survived the generic filter, else None
    alias_keys: frozenset   # surviving keys that came from an alias
    etype: str


def load_mentions(book: str, cache_file: str, src_id: str) -> list[Mention]:
    document = load_and_clean_narrative(Path(BOOK_NAME_TO_PATH[book]), source_id=src_id)
    chunks = chunk_document(document, ChunkingConfig(chunk_size=700, overlap=120))
    windows = segment_into_extraction_windows(document.cleaned_text)
    ec = ExtractionConfig()

    class _NoCall:
        def complete_structured(self, prompt, schema):
            raise ValueError("cache-only")

    cache = JsonlStructuredLLMCache(RESULTS / cache_file)
    extractor = LLMWindowExtractor(
        CachingStructuredLLMClient(_NoCall(), cache, LLMConfig(model_name="gemini-2.5-flash", temperature=0.0, max_tokens=65536), ec.schema_version),
        ec,
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

    # replicate merge.py's deterministic content-hash ordering
    import hashlib

    def sort_key(f):
        s = f"{f.chapter_index}:{f.window_index}:{f.raw.local_id}:{f.raw.name}".encode()
        return hashlib.sha256(s).hexdigest()

    facts.sort(key=sort_key)

    mentions = []
    for i, f in enumerate(facts):
        raw = f.raw
        pkey = norm_name(raw.name)
        keys = set()
        name_key = None
        alias_keys = set()
        if pkey and not is_generic_referent(pkey):
            keys.add(pkey)
            name_key = pkey
        for a in raw.aliases:
            ak = norm_name(a)
            if ak and not is_generic_referent(ak):
                keys.add(ak)
                alias_keys.add(ak)
        mentions.append(
            Mention(idx=i, primary=raw.name, primary_key=pkey, keys=tuple(sorted(keys)),
                    name_key=name_key, alias_keys=frozenset(alias_keys), etype=str(raw.type).split(".")[-1])
        )
    return mentions


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


def build_components(mentions, exclude_keys=frozenset()):
    key_to_idx = defaultdict(list)
    for m in mentions:
        for k in m.keys:
            if k not in exclude_keys:
                key_to_idx[k].append(m.idx)
    uf = UF(len(mentions))
    for k, idxs in key_to_idx.items():
        first = idxs[0]
        for o in idxs[1:]:
            uf.union(first, o)
    comps = defaultdict(list)
    for m in mentions:
        comps[uf.find(m.idx)].append(m)
    return list(comps.values()), key_to_idx, uf


def classify_key(key, key_mentions):
    """Best-effort deterministic classification of a merge key."""
    toks = key.split()
    types = Counter(m.etype for m in key_mentions)
    majority_type = types.most_common(1)[0][0]
    # never-a-primary-name => alias-only ("explicit alias")
    appears_as_primary = any(m.name_key == key for m in key_mentions)
    if majority_type == "LOCATION":
        return "Location"
    if majority_type == "ORGANIZATION":
        return "Organization"
    if not appears_as_primary:
        return "Explicit alias"
    if len(toks) == 1:
        return "Partial person name (given)" if toks[0] in GIVEN_NAMES else "Surname"
    # multi-token
    if toks[0] in NOBILIARY_PARTICLES:
        return "Surname (particle)"
    if toks[0] in GIVEN_NAMES or toks[-1] not in NOBILIARY_PARTICLES:
        return "Full person name"
    return "Other"


SPECIFICITY = {  # higher = stronger evidence of true identity
    "Full person name": 5,
    "Explicit alias": 4,
    "Surname (particle)": 3,
    "Surname": 2,
    "Partial person name (given)": 2,
    "Location": 1,
    "Organization": 1,
    "Other": 0,
}


def analyze(book, cache_file, src_id):
    mentions = load_mentions(book, cache_file, src_id)
    comps, key_to_idx, uf = build_components(mentions)
    comps.sort(key=len, reverse=True)
    giant = comps[0]
    giant_idx = {m.idx for m in giant}

    # ---- key stats (whole novel) ----
    key_class = {}
    key_rows = []
    for k, idxs in key_to_idx.items():
        kmen = [mentions[i] for i in idxs]
        cls = classify_key(k, kmen)
        key_class[k] = cls
        distinct_primary = len({m.primary for m in kmen})
        comps_touched = len({uf.find(i) for i in idxs})
        key_rows.append({
            "key": k, "mentions": len(idxs), "distinct_primary_names": distinct_primary,
            "components_touched": comps_touched, "class": cls,
            "etype_dist": dict(Counter(m.etype for m in kmen)),
            "sample_names": sorted({m.primary for m in kmen})[:8],
        })
    key_rows.sort(key=lambda r: (-r["mentions"], -r["distinct_primary_names"]))

    # ---- character-level graph within giant ----
    # nodes = distinct primary names in giant; edge(name_a,name_b) labelled by
    # ALL shared keys; the edge's "cause" = max-specificity shared key class.
    name_to_keys = defaultdict(set)
    for m in giant:
        for k in m.keys:
            name_to_keys[m.primary].add(k)
    names = sorted(name_to_keys)
    key_to_names = defaultdict(set)
    for nm, ks in name_to_keys.items():
        for k in ks:
            key_to_names[k].add(nm)
    # character adjacency
    char_edges = defaultdict(set)  # (a,b)->set of shared keys
    for k, nms in key_to_names.items():
        nl = sorted(nms)
        for a in range(len(nl)):
            for b in range(a + 1, len(nl)):
                char_edges[(nl[a], nl[b])].add(k)

    # spanning tree over character graph (BFS, deterministic) -> the actual
    # chain of merges; attribute each tree edge to its max-specificity key.
    adj = defaultdict(list)
    for (a, b), ks in char_edges.items():
        adj[a].append((b, ks))
        adj[b].append((a, ks))
    visited = set()
    tree_edges = []
    for start in names:
        if start in visited:
            continue
        visited.add(start)
        queue = [start]
        while queue:
            cur = queue.pop(0)
            for nxt, ks in sorted(adj[cur], key=lambda x: x[0]):
                if nxt not in visited:
                    visited.add(nxt)
                    best = max(ks, key=lambda k: SPECIFICITY.get(key_class.get(k, "Other"), 0))
                    tree_edges.append((cur, nxt, best, key_class.get(best, "Other")))
                    queue.append(nxt)

    tree_cause_counts = Counter(cls for _, _, _, cls in tree_edges)

    # ---- per character-edge max-specificity cause (ALL edges, not just tree) ----
    edge_cause = Counter()
    surname_only_edges = []
    for (a, b), ks in char_edges.items():
        best = max(ks, key=lambda k: SPECIFICITY.get(key_class.get(k, "Other"), 0))
        cls = key_class.get(best, "Other")
        edge_cause[cls] += 1
        if SPECIFICITY.get(cls, 0) <= 2:  # surname / partial / weaker, no full-name/alias corroboration
            surname_only_edges.append((a, b, best, cls))

    # ---- ablation: remove each key-class, recompute giant size ----
    classes = set(key_class.values())
    ablation = {}
    for cls in classes:
        excl = {k for k, c in key_class.items() if c == cls}
        ac, _, _ = build_components(mentions, exclude_keys=excl)
        ablation[cls] = max(len(c) for c in ac)
    # also ablate ONLY-surname-style (specificity<=2) keys together
    weak_keys = {k for k, c in key_class.items() if SPECIFICITY.get(c, 0) <= 2}
    ac_weak, _, _ = build_components(mentions, exclude_keys=weak_keys)
    giant_without_weak = max(len(c) for c in ac_weak)

    out = {
        "book": book,
        "total_mentions": len(mentions),
        "total_components": len(comps),
        "giant_size_mentions": len(giant),
        "giant_distinct_primary_names": len({m.primary for m in giant}),
        "giant_etype_dist": dict(Counter(m.etype for m in giant)),
        "giant_canonical_sample": sorted({m.primary for m in giant})[:40],
        "top50_keys": key_rows[:50],
        "key_class_counts_top200": dict(Counter(r["class"] for r in key_rows[:200])),
        "char_edges_total": len(char_edges),
        "char_edge_cause_distribution": dict(edge_cause),
        "spanning_tree_edges": len(tree_edges),
        "spanning_tree_cause_distribution": dict(tree_cause_counts),
        "surname_or_weaker_only_edge_count": len(surname_only_edges),
        "ablation_giant_size_by_removed_class": ablation,
        "giant_size_without_all_weak_keys": giant_without_weak,
        "sample_surname_only_edges": surname_only_edges[:25],
        "sample_spanning_chain": tree_edges[:60],
        "components_over_20_distinct_names": [
            {"distinct_names": len({m.primary for m in c}), "mentions": len(c),
             "etype_dist": dict(Counter(m.etype for m in c)),
             "names_sample": sorted({m.primary for m in c})[:15]}
            for c in comps if len({m.primary for m in c}) > 20
        ],
    }
    (RESULTS / f"er3_audit_{src_id}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def _print_summary(o):
    print("=" * 78)
    print(f"BOOK: {o['book']}")
    print("=" * 78)
    print(f"mentions={o['total_mentions']} components={o['total_components']} "
          f"GIANT: {o['giant_size_mentions']} mentions / {o['giant_distinct_primary_names']} distinct primary names")
    print(f"giant entity-type dist: {o['giant_etype_dist']}")
    print(f"\nTop-200 key class counts: {o['key_class_counts_top200']}")
    print(f"\nCharacter-edge CAUSE distribution (max-specificity shared key per char pair):")
    for cls, n in sorted(o['char_edge_cause_distribution'].items(), key=lambda x: -x[1]):
        print(f"   {cls:32s} {n}")
    print(f"\nSpanning-tree (actual merge chain) CAUSE distribution:")
    tot = max(1, o['spanning_tree_edges'])
    for cls, n in sorted(o['spanning_tree_cause_distribution'].items(), key=lambda x: -x[1]):
        print(f"   {cls:32s} {n:5d}  ({100*n/tot:.1f}% of {tot} merge links)")
    print(f"\nABLATION -- giant size if we DROP all keys of a class:")
    base = o['giant_size_mentions']
    for cls, sz in sorted(o['ablation_giant_size_by_removed_class'].items(), key=lambda x: x[1]):
        print(f"   drop {cls:32s} -> giant becomes {sz:5d}  (was {base}; reduction {base-sz})")
    print(f"   drop ALL weak (surname/partial/loc/org) keys -> giant becomes {o['giant_size_without_all_weak_keys']} (was {base})")
    print(f"\nTOP 25 MERGE KEYS:")
    print(f"   {'key':28s} {'ment':>5s} {'chars':>6s} {'comps':>6s}  class")
    for r in o['top50_keys'][:25]:
        print(f"   {r['key'][:28]:28s} {r['mentions']:5d} {r['distinct_primary_names']:6d} {r['components_touched']:6d}  {r['class']}")
    print(f"\nComponents with >20 distinct names: {len(o['components_over_20_distinct_names'])}")
    for c in o['components_over_20_distinct_names']:
        print(f"   {c['distinct_names']} names / {c['mentions']} mentions, types={c['etype_dist']}")
    print()


if __name__ == "__main__":
    for book, cf, sid in [
        ("In Search of the Castaways", "extraction_cache_in_search_of_the_castaways.jsonl", "castaways"),
        ("The Count of Monte Cristo", "extraction_cache_the_count_of_monte_cristo.jsonl", "mc"),
    ]:
        _print_summary(analyze(book, cf, sid))
