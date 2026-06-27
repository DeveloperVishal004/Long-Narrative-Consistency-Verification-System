# LNCVS — Long Narrative Consistency Verification System

Verify whether a narrative claim is consistent with a long source narrative
(50,000–100,000+ words) — and get a deterministic, auditable verdict, never an
LLM's guess.

LNCVS is a **consistency verification system, not a question-answering system.** Given
a `(source narrative, narrative claim)` pair, it returns one of three verdicts, plus
the supporting evidence, the contradicting evidence, a full evidence trace, and a
reasoning summary — all carried in a single typed `EvidenceLedger`.

```
Input :  (source narrative, narrative claim)
Output:  CONSISTENT | CONTRADICTORY | INSUFFICIENT_EVIDENCE
         + supporting evidence + contradicting evidence + evidence trace
```

## Why this exists

Large-context LLMs are surprisingly bad at long-narrative consistency. Simply feeding
a 100k-word novel and a claim into a bigger context window does not solve the dominant
failure modes — the long context fallacy, attention dilution, implicit memory,
constraint loss, plausibility bias, and retrieval failure.

The central hypothesis behind LNCVS is that **graph retrieval + semantic retrieval +
lexical retrieval + explicit evidence tracking** beats single-pass RAG for this task,
because the verdict is produced by **deterministic Python over an auditable evidence
ledger** rather than by a model.

## The three verdicts

| Verdict | Meaning |
|---|---|
| `CONSISTENT` | Every atomic claim is entailed by retrieved evidence (strict policy), or no atomic claim is contradicted (lenient policy). |
| `CONTRADICTORY` | At least one atomic claim is contradicted by retrieved evidence. |
| `INSUFFICIENT_EVIDENCE` | At least one atomic claim has neither entailing nor contradicting evidence — an explicit coverage gap, not a contradiction. |

Verdicts are **deterministic and reproducible**: the same `(narrative, claim)` pair
under the same configuration always yields the same verdict.

## Design principles

These are the non-negotiable rules the codebase obeys:

- **The rule engine, never an LLM, produces the verdict.** LLMs may decompose, score,
  and verify; only deterministic Python maps the ledger to a `FinalVerdict`. This is
  the cardinal invariant.
- **Determinism by construction** — pinned seeds/temperatures, content-hash IDs,
  input-hash caching, and config fingerprints. Determinism is enforced, not hoped for.
- **Vertical slice before breadth** — a correct end-to-end path worked before any
  second retriever or representation was added.
- **Explainability over cleverness** — every verdict traces to specific evidence
  chunks via the ledger.
- **Typed boundaries** — no raw `dict` or `List[Any]` ever crosses a module boundary;
  closed vocabularies are enums.
- **Never silently fail** — a swallowed exception and a config field read nowhere are
  equally forbidden.
- **One SDK per file** — each third-party SDK is confined to exactly one module to
  contain the blast radius of any vendor change.

## Architecture

LNCVS has **two build-time pipelines** and **one query-time pipeline**, all converging
on a single `EvidenceLedger`.

```
BUILD-TIME (per novel, once, cached)        QUERY-TIME (per claim)
─────────────────────────────────────       ──────────────────────────────────────
raw narrative                                narrative claim
   │ ingestion (load + clean)                   │ decomposition  → atomic claims (LLM)
   │ chunking (sliding window, hash IDs)        │ question gen   → probe questions (LLM)
   ├─► ChromaIndex   (dense vectors)            │ query builder
   ├─► BM25Index     (lexical)                  │ retrieval orchestrator ◄── indices
   └─► GraphIndex    (entities, opt-in)         │ RRF fusion
                                                │ fact verification / NLI
                                                │ classify() → per-claim status
                                                └─► ThresholdRuleEngine → FinalVerdict

                    └────────────► EvidenceLedger (single source of truth) ◄────────────┘
```

### Two pipelines, one important distinction

- **Production / canonical pipeline** (`orchestration.LangGraphPipeline`): the linear
  vertical slice — semantic + BM25 retrieval → RRF fusion → cross-encoder NLI → rule
  engine. Graph-free, cross-encoder-based.
- **Research / experimental pipeline** (`scripts/evaluate_with_graph.py`): the same
  linear pipeline **plus** a `GraphRetriever` source **and** an LLM-backed fact
  verifier swapped in for the cross-encoder.

> ⚠️ **The knowledge graph is opt-in.** It is wired only in the evaluation script and
> was never promoted into the production LangGraph node path.

## Repository structure

```
src/lncvs/
├── schemas/        # ALL shared Pydantic models + enums — the universal leaf
├── llm/            # provider-agnostic LLM protocols, configs, caches, Gemini/OpenAI clients
├── ingestion/      # raw load + offset-preserving cleaning
├── chunking/       # sliding-window chunking, content-hash IDs, span-overlap helpers
├── indexing/       # Embedder, ChromaIndex, BM25Index, tokenizer, caches
├── retrieval/      # Retriever protocol, Semantic/BM25 retrievers, orchestrator, query builder
├── fusion/         # Reciprocal Rank Fusion (+ fingerprinted config)
├── reasoning/
│   ├── decomposition/      # claim → atomic claims (LLM)
│   ├── questions/          # atomic claim → probe questions (LLM)
│   ├── nli/                # cross-encoder NLI model + verifier + cache
│   └── fact_verification/  # FactVerifier protocol, CrossEncoder + LLM verifiers
├── ledger/         # EvidenceLedger mutation API (LedgerService)
├── rules/          # RuleEngine ABC, pure classify(), ThresholdRuleEngine
├── orchestration/  # LangGraph StateGraph, nodes, reducers, channels, resources
├── evaluation/     # PipelineRunner, EvaluationHarness, metrics, reporting, dataset
├── graph/          # (opt-in) segmentation, llm_extraction, provenance, entity_resolution,
│                   #   construction, builder/index/traversal/retriever
├── cli/            # NOT IMPLEMENTED (empty package, documented gap)
└── configs/        # NOT IMPLEMENTED (empty package)

scripts/   # evaluation drivers, audits, prototypes, diagnostics (the de-facto interface)
tests/     # 114 test modules
data/      # the two real novels + train/test CSVs + small synthetic narratives
datasets/  # gold JSONL
results/   # curated run reports (large caches/artifacts are git-ignored)
```

Dependencies point downward only:
`ingestion/chunking/indexing → retrieval → fusion → reasoning → ledger → rules → orchestration → evaluation`.
`schemas/` is the universal leaf (everything may depend on it; it depends on nothing).

## Installation

Requires **Python 3.11+**.

```bash
git clone https://github.com/DeveloperVishal004/Long-Narrative-Consistency-Verification-System.git
cd Long-Narrative-Consistency-Verification-System
pip install -e .
```

Core dependencies (from `pyproject.toml`): `pydantic>=2.0`,
`sentence-transformers>=2.2`, `chromadb>=0.4`, `rank-bm25>=0.2`, `numpy>=1.24`,
`matplotlib>=3.7`, `langgraph>=1.2,<2.0`, `networkx>=3.0`, `tiktoken>=0.7`,
`openai>=1.0`, `google-genai>=1.0`, `python-dotenv>=1.0`.

### API keys

LLM-backed steps (claim decomposition, question generation, graph extraction, the LLM
fact verifier) need provider credentials. Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
# then edit .env:
GEMINI_API_KEY=...      # used by the Gemini clients
OPENAI_API_KEY=...      # used by the OpenAI structured client / LLM fact verifier
```

The JSONL caches under `results/` are committable. Once populated, decompositions and
graph extractions rebuild with **zero API calls** — the "Tier-1 reproducibility
guarantee." If you have the caches, you can reproduce runs without any keys.

## Usage

> ⚠️ **There is no CLI yet** (`cli/` is an empty, documented package). The de-facto
> interface is the `scripts/` directory.

### Run the graph-impact / verifier evaluation

```bash
python scripts/evaluate_with_graph.py
```

Runs the experimental pipeline (linear + graph retriever + selectable verifier) over
the dataset and writes fingerprinted reports to `results/`.

### Validate at long-narrative scale

```bash
python scripts/validate_long_narrative.py
```

Executes the real LangGraph pipeline end-to-end on a full-length novel and checks for
NLI truncations, peak memory, and determinism.

### Diagnostics

```bash
python scripts/diagnose_retrieval_recall.py        # is retrieval the bottleneck?
python scripts/diagnose_missed_contradictions.py   # why were contradictions missed?
python scripts/audit_er3_merge_keys.py             # entity-resolution merge-key audit
```

## Key configuration

| Config | Field | Production value |
|---|---|---|
| `ChunkingConfig` | chunk_size / overlap | 700 / 120 (keeps NLI premise+hypothesis under max_length=256) |
| `EmbeddingConfig` | model_name | all-MiniLM-L6-v2, L2-normalized, CPU |
| `RetrievalConfig` | top_k | 10 per query |
| `FusionConfig` | rrf_k / top_k_fused | 60 / 10 |
| `NLIConfig` | model_name | cross-encoder/nli-deberta-v3-base, max_length=256 |
| `RuleEngineConfig` | contradiction_threshold | 0.9 (raised from 0.5) |
| `RuleEngineConfig` | consistency_requires_entailment | False (lenient) in real runs |
| Decomposition / extraction | model | gemini-2.5-flash, temperature 0 |
| LLM fact verifier | model | gpt-4o-2024-08-06, temperature 0 |

## How a claim flows through the system

Using the canonical test case — narrative *"John lost his left arm in an accident in
2010. John moved to London in 2012."*, claim *"John played a complex two-handed piano
piece at a pub in London."* (expected: `CONTRADICTORY`):

1. **Decomposition** → atomic claims ("John played piano", "John used both hands",
   "the event occurred in London").
2. **Question generation** → probe questions ("Did John lose an arm?").
3. **Retrieval** → semantic + BM25 (+ optional graph) evidence, claim-stamped at the
   orchestrator.
4. **Fusion** → Reciprocal Rank Fusion combines sources on **rank, not score**
   (BM25 is unbounded; cosine is `[0,1]` — incomparable).
5. **Verification** → NLI / fact verification flags "lost his left arm" as
   contradicting "used both hands."
6. **Classification + rule engine** → a contradicted claim fires Rule 1, yielding
   `CONTRADICTORY`.

Every step records into the single `EvidenceLedger` via `LedgerService`, appending to
an append-only audit log.

## Testing

```bash
pytest                    # default: excludes slow (real-book) tests
pytest -m slow            # the long-narrative validation tests
```

114 test modules. First-class **determinism tests** include chunk-ID stability,
evidence-ID determinism, byte-identical `ledger_fingerprint` equivalence between the
LangGraph pipeline and the `PipelineRunner` oracle, and full-pipeline N-run
reproducibility. Silently-reversible logic (NLI premise/hypothesis direction, label
mappings, graph relation direction) is guarded by pinned regression tests.

## Results

On `data/train.csv` (n=80; ~64% / 36% consistent / contradict). A trivial
always-`CONSISTENT` classifier scores ~64% — the baseline for reading every number
below.

| Condition | Accuracy | Macro-F1 | CONTRADICTORY recall |
|---|---|---|---|
| Cross-encoder verifier | 0.359 | 0.176 | 1.00 (predicts CONTRADICTORY for everything) |
| LLM verifier (baseline) | 0.690 | 0.340 | 0.12 |
| LLM verifier (with graph) | 0.671 | 0.315 | 0.08 |

Key findings:

- **The verifier was the dominant bottleneck.** Swapping the cross-encoder for the
  evidence-set LLM verifier lifted accuracy by ~33 points with zero retrieval/graph
  work.
- **The opt-in graph changed 0/80 verdicts** under the cross-encoder, and slightly
  perturbed (within noise) the LLM verifier.
- **The remaining ceiling is contradiction detection.** The missed contradictions are
  "fabricated-detail-vs-silence" cases (an invented date/ritual the source is silent
  on) — a deliberate consequence of the `INSUFFICIENT_EVIDENCE`-never-`CONTRADICTORY`
  design, not a retrieval bug.

## Limitations (verifiable, disclosed)

- **No CLI** — `cli/` and `configs/` are empty packages (a spec-vs-implementation gap).
- **Verdict ceiling is contradiction detection** — CDR ~0.08–0.12; accuracy only ~5
  points above the trivial baseline.
- **Graph retrieval is opt-in** and never promoted into the LangGraph node path.
- **Graph entry resolution is exact-match only** — correctly-merged non-winning
  aliases are unreachable by query.
- **Events are constructed but disabled** on the extraction wire (cost) and never
  reach retrieval.
- **No index persistence** — Chroma/BM25 are ephemeral and rebuilt each run.
- **Residual entity over-merge** from genuine LLM extraction errors and source-text
  naming overlap (out of scope for the conservative no-fuzzy policy).
- **Probe questions are not exercised** in the graph-impact study.

## Roadmap

Temporal reasoning and event-timeline graphs; event-aware multi-hop traversal; graded
consistency scoring beyond the ternary verdict; NER/coreference/linking for entity and
event extraction; index persistence; a real CLI; and promoting graph retrieval into
the LangGraph node path as a measured ablation.

> **The V2 entry gate:** every graph component must demonstrate measurable improvement
> over the Chroma+BM25 baseline before it is adopted into production.

## Documentation

The complete reverse-engineered technical reference — the engineering handbook — lives
in `MASTER_TECHNICAL_REFERENCE_PART1.md` through `PART4.md`. The governing
specifications are `Project_spec.md` (what to build) and `CLAUDE.md` (how to build).
