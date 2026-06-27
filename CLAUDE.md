# CLAUDE.md

## Project Overview

Project Name: Long Narrative Consistency Verification System (LNCVS)

**Source of truth: `PROJECT_SPEC.md`.** Wherever this file and `PROJECT_SPEC.md` appear to disagree, `PROJECT_SPEC.md` wins. This file governs *how* we build; the spec governs *what* we build.

### What this system is

LNCVS verifies whether a **narrative claim** is consistent with a **long source narrative** (50,000–100,000+ words). It is a **consistency verification system, not a question-answering system.**

Input: `(source narrative, narrative claim)`
Output: a deterministic verdict, supporting evidence, contradicting evidence, an evidence trace, and a reasoning summary.

### What this system is not

- Not a general-purpose RAG/QA assistant.
- Not a system that answers open-ended questions about a document.
- Not a system where an LLM produces the final decision.

### Final Verdicts

The system produces exactly one of three verdicts per claim:

- `CONSISTENT` — every atomic claim is entailed by retrieved evidence.
- `CONTRADICTORY` — at least one atomic claim is contradicted by retrieved evidence.
- `INSUFFICIENT_EVIDENCE` — at least one atomic claim has no entailing or contradicting evidence. This is **not** a contradiction; it is an explicit admission of a retrieval/coverage gap.

Verdicts must be deterministic and reproducible: the same `(narrative, claim)` pair, evaluated with the same configuration, must always produce the same verdict.

---

## Development Philosophy

Before writing code:

1. Think deeply about the problem.
2. Propose architecture first.
3. Explain tradeoffs.
4. Identify failure modes.
5. Only then implement.

Never immediately generate code without design reasoning.

Additional philosophy specific to this project:

- **Vertical slice before breadth.** Version 1 must reach a correct, working end-to-end path (ingestion → retrieval → NLI → verdict) before additional retrieval mechanisms or representations are added. Do not build the Knowledge Graph, BM25, or Question Generation before the single-retriever slice produces a correct verdict on the dummy test case.
- **Determinism is a requirement, not an aspiration.** Any component with non-deterministic behavior (LLM calls, embedding model upgrades) must be made deterministic by construction — pin seeds/temperature, and cache outputs by input hash.
- **The rule engine, never an LLM, produces the verdict.** This is non-negotiable. LLMs may decompose claims, generate questions, or score entailment; only deterministic Python code maps the Evidence Ledger to a `FinalVerdict`.
- **Explainability over cleverness.** Every verdict must be traceable to specific evidence chunks via the Evidence Ledger. If a design choice makes the reasoning trace harder to audit, prefer the less clever alternative.
- Always prefer maintainability over cleverness. Code should be understandable by a new contributor.

---

## Technical Stack

Language:

- Python 3.11+

Core Libraries:

- LangGraph
- LangChain
- FAISS / ChromaDB
- rank-bm25
- Pydantic v2
- FastAPI (deferred — see Version 1 Scope)
- Transformers
- Sentence Transformers

Version 2 only:

- NetworkX (Knowledge Graph construction and traversal)

Development Tools:

- pytest
- black
- ruff
- mypy

Visualization:

- matplotlib
- plotly

Storage:

- Local filesystem initially
- ChromaDB for dense vectors
- BM25 corpus persisted alongside Chroma, sharing the same chunk-ID space
- Neo4j or graph persistence — Version 2 only, not before

---

## Architecture Principles

Follow these principles:

1. Modular design
2. Separation of concerns
3. Dependency injection
4. Testability
5. Reproducibility

Avoid:

- God classes
- Circular imports
- Hidden state
- Global variables
- Untyped dictionaries crossing module boundaries

### Dependency rules

- `schemas` is the universal leaf module. It depends on nothing. Every other module may depend on it.
- No module may import another module's internals. Cross-module communication happens only through `schemas` types and exposed service interfaces.
- Dependencies point downward only: `ingestion/chunking/indexing` → `retrieval` → `fusion` → `reasoning` → `ledger` → `rules` → `orchestration` → `evaluation`. Nothing upstream may import something downstream.
- `llm` is a second, narrow leaf module (depends only on nothing besides the standard library): it provides the provider-agnostic `LLMClient` protocol, `LLMConfig`, and the caching decorator. `reasoning/decomposition` and (later) `reasoning/questions` depend on `llm`; `ledger` must never import from `reasoning/decomposition` — ID derivation happens in the decomposer, and the ledger only stores the IDs it's handed.

---

## Repository Structure

```
lncvs/
├── configs/                 # settings, profiles (dev/test/eval)
├── schemas/                 # ALL shared Pydantic models — the data contracts
├── llm/                       # LLMClient protocol, LLMConfig, LLM completion cache (no vendor SDK)
├── ingestion/                # raw narrative loading + cleaning
├── chunking/                 # chunk + metadata + deterministic IDs
├── indexing/                 # Chroma + BM25 index build/load
├── retrieval/                # Retriever interfaces + hybrid orchestration
├── fusion/                    # RRF fusion + dedup
├── reasoning/
│   ├── decomposition/        # claim → atomic claims
│   ├── questions/             # atomic claim → probe questions
│   └── nli/                   # NLI verification
├── ledger/                    # EvidenceLedger model + mutation API
├── rules/                     # deterministic rule engine
├── orchestration/             # LangGraph wiring + GraphState
├── evaluation/                 # metrics + harness + datasets
├── cli/                         # command-line entrypoint
├── tests/
├── datasets/                   # synthetic + fixtures
├── notebooks/
└── docs/                         # ADRs, design reviews

# Version 2 only — do not create in Version 1:
├── extraction/                # entity & event extraction
├── graph/                      # Knowledge Graph construction + Graph Retrieval
```

Every module should have:

- `__init__.py`
- `config.py`
- `models.py` (re-exports from `schemas`; does not define competing types)
- `service.py`
- `tests/`

There is no `frontend/` module in Version 1. The interface is the CLI.

---

## Coding Standards

Requirements:

- Full type hints
- Docstrings for public methods
- Meaningful variable names
- No magic numbers
- Small functions

Maximum function size: ~50 lines preferred.
Maximum class size: ~300 lines preferred.

Break large logic into helper methods.

### Pydantic Standards

All schemas must use:

- Pydantic v2
- Field validation
- Clear descriptions

**Never pass dictionaries or `List[Any]` between modules.** This includes the Evidence Ledger — every field must be a typed model, never a raw dict. This is a direct, non-negotiable consequence of `PROJECT_SPEC.md`'s explainability requirement: an untyped ledger cannot be audited.

Closed vocabularies must be enums, not strings: NLI labels, retrieval sources, verdicts, entity/edge types (V2).

Required core models (defined in `schemas/`, not duplicated elsewhere):

- `DocumentChunk`
- `AtomicClaim`
- `ProbeQuestion`
- `RetrievedEvidence`
- `FusedEvidence`
- `NLIResult`
- `Contradiction`
- `SupportingEvidence`
- `ReasoningStep`
- `LedgerEvent`
- `EvidenceLedger`
- `FinalVerdict`
- `ControlState`
- `GraphState`

(Version 2 adds: `EntityRecord`, `EventRecord`, `GraphNode`, `GraphEdge`, `Provenance` extensions for graph edges.)

### LLM and Embedding Determinism

Any component that calls an LLM or embedding model must be wrapped in its
caching decorator (`CachingLLMClient` for `llm/`, `CachingEmbedder` for
`indexing/`) in the same change that introduces it — not as a follow-up.

The cache key is `(config.fingerprint(), input_text)`. `LLMConfig.fingerprint()`
covers only model identity and sampling parameters (model name, temperature,
max_tokens) — it deliberately does **not** fold in a prompt-template
version, because the cache key already includes the full rendered prompt
text. A prompt-template edit changes that text, which changes the cache key
automatically; no separate version-bumping discipline is needed for cache
correctness. A `prompt_version` field may still exist on a config object
(e.g. `DecompositionConfig`) as **audit provenance** — recording which
template produced a given result — but it must never be treated as a
substitute for including the rendered prompt itself in the cache key.

Content-hash IDs (chunk IDs, evidence IDs, atomic claim IDs, probe question
IDs) follow the same principle as caching: derive the ID from the content,
never from `uuid4()` or another source of randomness, so identical input
always produces an identical ID across runs and across process instances.

Wall-clock fields (`ReasoningStep.timestamp`, `LedgerEvent.timestamp`) are
the one accepted exception to full ledger reproducibility — the rule engine
never reads them, so they may differ between runs. Determinism is required
for claim content, evidence content, and the final verdict; it is not
required for audit-log timestamps.

### Question Generation: Empty Results Are Valid

Unlike claim decomposition (where zero atomic claims is a parser error —
a claim must decompose into *something*), question generation may
legitimately produce **zero** probe questions for an atomic claim. Retrieval
always has the claim text itself as a fallback query, so probe questions are
strictly additive coverage, not a required input. `parse_question_response`
returns `[]` in this case rather than raising; only malformed LLM output
(bad JSON, wrong shape) raises.

Probe questions are filtered to those structurally phrased as questions
(ending in `?`) before being returned. This is a deliberately weak,
non-semantic proxy for "the question must ask about the claim's subject
rather than assert a new, unstated fact as already true" — it catches the
clearest failure mode (a declarative statement returned instead of a
question) but is not a substitute for true faithfulness checking, which
requires an LLM-judge evaluation deferred to a later phase. A generated
question may, by design, mention something absent from the claim (e.g. the
claim "John used both hands" legitimately probes "Did John lose an arm?")
— faithfulness here means not *asserting* unstated facts, not avoiding
unstated *topics*.

### Retrieval Orchestration: Claim-Agnostic Retriever, Provenance at the Boundary

`Retriever` and `Indexer` (and their implementations, e.g. `SemanticRetriever` /
`ChromaIndex`) are and must remain **claim-agnostic**: they answer "given
this query text, which chunks?" and know nothing about `AtomicClaim`,
`ProbeQuestion`, or `RetrievalQuery`. `RetrievedEvidence.atomic_claim_id`
and `.query_id` are therefore Optional at construction time — a bare
`Retriever` call cannot supply them.

`RetrievalOrchestrator` is the one layer that knows about claims and
queries. It takes a **list** of `Retriever`s (one per source — semantic,
lexical, and any future backend), runs each `RetrievalQuery` through every
one of them in fixed order, and stamps `atomic_claim_id` and `query_id`
onto each result via `model_copy` (the underlying `RetrievedEvidence` is
frozen). **The linkage invariant — "every piece of evidence in the ledger
is claim-linked" — is enforced at the ledger write boundary**
(`LedgerService.record_retrieved_evidence`), not at the type level. Never
call `add_retrieved_evidence`/extend the ledger directly with evidence that
hasn't passed through the orchestrator. A single-retriever list is a valid,
fully-supported degenerate case (semantic-only retrieval behaves exactly as
before hybrid retrieval was added).

**`evidence_id` must be re-derived from `(query_id, source)`, not raw query
text or query_id alone.** `evidence_id = hash(query_text, chunk_id, rank)`
(a bare retriever's own derivation) collides when two different atomic
claims happen to generate identical query text — fixed by folding in
`query_id`, which already encodes `(atomic_claim_id, origin, question_id,
text)`. A second collision opens up once a second retrieval backend exists:
two different sources can return the **same chunk at the same rank for the
same query** (e.g. semantic and lexical both rank a chunk #1), which would
collide under `hash(query_id, chunk_id, rank)` alone. `RetrievalOrchestrator`
therefore re-derives `evidence_id = hash(query_id, source, chunk_id, rank)`
when stamping — collision-free across claims, queries, *and* sources.

A `RetrievalQuery` with zero results (from one retriever, or all of them)
is not an error — it is the correct input to `INSUFFICIENT_EVIDENCE`. Do
not raise on empty retrieval results; let them flow through to the rule
engine.

### Hybrid Retrieval: Shared Tokenization and BM25 Index Design

`BM25Index` implements `Indexer` exactly as `ChromaIndex` does, and is the
only module that imports `rank_bm25` — the same isolation discipline
applied to `chromadb`. It shares the same chunk-ID space as `ChromaIndex`
by construction: both index identical `DocumentChunk` lists with
deterministic content-hash IDs, so fusion can dedup by `chunk_id` across
sources with no ID-reconciliation step.

**Corpus and query text must be tokenized by the exact same function.**
`lncvs.indexing.tokenizer.tokenize()` is that single function, used on both
sides of `BM25Index`. Tokenization divergence between indexing and
querying is a silent recall killer — a query token that doesn't match the
corpus's tokenization scheme never retrieves anything, with no error to
signal it. Never introduce a second tokenizer for either side.

### Fusion: Reciprocal Rank Fusion, Minimal FusedEvidence, Deterministic Ties

RRF operates on **ranks, not raw scores** — this is why it was chosen over
score normalization: `RetrievedEvidence.raw_score` is explicitly
incomparable across sources (BM25 scores are unbounded; cosine similarity
is bounded [0,1]), but rank position is always comparable. `fuse_evidence`
groups claim-linked evidence by `(atomic_claim_id, chunk_id)` and sums
`1/(rrf_k + rank)` over every `(query, source)` contribution.

**`FusedEvidence` is deliberately minimal**: `atomic_claim_id`, `chunk_id`,
`text`, `rrf_score`, `contributing_sources`, `contributing_query_ids` — and
explicitly **no `source_ranks` or other per-source rank field**. Per-rank
detail already lives, undeduplicated and unambiguous, in
`EvidenceLedger.retrieved_evidence`; denormalizing a summarized version
onto `FusedEvidence` would risk drifting from that source of truth for no
offsetting benefit, and "one rank per source" is lossy the moment a claim
has more than one query per source (which is the common case — a claim
query plus N probe questions). If you need a chunk's per-source ranks,
join `retrieved_evidence` on `(atomic_claim_id, chunk_id)`.

**`FusionConfig` must be fingerprinted, like every other config in this
system** (`EmbeddingConfig`, `LLMConfig`). Given the ledger's
`retrieved_evidence` plus a recorded `FusionConfig.fingerprint()`, any
`FusedEvidence.rrf_score` must be independently recomputable and
verifiable — this fingerprint is what replaces the audit value a
denormalized `source_ranks` field would have offered.

**RRF ties must break deterministically.** Equal `rrf_score` for two
chunks within the same claim resolves by ascending `chunk_id`, never by
incidental dict/list iteration order. `fusion/` imports `schemas` only —
never `lncvs.retrieval` — keeping it a pure, independently-testable layer
that transforms schema types without depending on how they were produced.

---

### NLI Verification and Verdict Construction: Evidence-Level NLI, Single Threshold Owner

`NLIVerifier` (`reasoning/nli/service.py`) is, and must remain,
**evidence-level only**: it returns one `NLIResult` per
`(AtomicClaim, FusedEvidence)` pair, with no claim-level aggregation and no
threshold application. The premise/hypothesis direction is fixed —
`premise = FusedEvidence.text`, `hypothesis = AtomicClaim.text` — and is
pinned by a direction-regression test, since this is silently reversible
and would invert every entailment/contradiction call if swapped.

**Threshold ownership is Design B, decided in the Phase 5 architecture
review: thresholds live in exactly one place, `RuleEngineConfig`, applied
by exactly one function, `lncvs.rules.classification.classify()`.**
`classify()` is a pure helper — `(nli_results, atomic_claim_ids, config) ->
ClassificationOutcome` — not an injected pipeline component. It is called
from two places that must always agree, and do so by construction because
the function is pure:
1. `ThresholdRuleEngine.evaluate(ledger)` — reads `ledger.nli_results` and
   `ledger.atomic_claims` directly, **never** the derived
   `contradictions`/`supporting_evidence`/`unsupported_claims` ledger
   fields, to avoid a circular "engine reads back its own derived output"
   path.
2. The Phase 5 driver — calls `classify()` once to get the
   `Contradiction`/`SupportingEvidence`/`unsupported_claim_ids` records,
   then writes them via `LedgerService.record_classification()` as an
   **explainability co-product**, never as verdict input.

**`CONTRADICTED` dominates `SUPPORTED` for the same claim** — a single
contradiction outweighs coexisting entailing evidence on that same claim.
This is not a fourth, ambiguous verdict state; the three-verdict model
(`CONSISTENT` / `CONTRADICTORY` / `INSUFFICIENT_EVIDENCE`) remains a
complete partition. A claim with **zero** `NLIResult`s (no fused evidence,
or none clearing either threshold) is `UNRESOLVED`, which routes to
`INSUFFICIENT_EVIDENCE` — never `CONTRADICTORY`. This is the cardinal
invariant of the whole rule engine and is tested explicitly at both the
`classify()` and `ThresholdRuleEngine` layers, and again end-to-end in the
Phase 5 acceptance test.

**`CrossEncoderNLIModel`'s label map is derived from the loaded model's own
`id2label`, never a hardcoded index order.** Different NLI checkpoints
order their output classes differently; trusting an assumed order is the
highest-risk silent failure mode in this module, because it would invert
every verdict without raising. `_build_label_map` reads `id2label` from the
model's own HuggingFace config and validates every label string against a
fixed alias set, raising loudly at construction time if a checkpoint's
labels can't be safely mapped.

**No `orchestration/` module was introduced in Phase 5.** The
fused-evidence → NLI → `classify()` → `ThresholdRuleEngine` wiring is thin
and lives directly in the Phase 5 acceptance test, per the Phase 5
architecture review's explicit instruction not to build a throwaway driver
that LangGraph (now Phase 8 — see the Evaluation Framework section below
for why Evaluation was promoted ahead of it) would immediately replace.

`CachingNLIModel` exists for parity with `CachingLLMClient`/
`CachingEmbedder`, but unlike those, a cross-encoder in eval mode with no
sampling is **already deterministic on its own** — the cache's value here
is performance (skipping redundant inference on repeated
`(premise, hypothesis)` pairs), not determinism. The determinism guarantee
for NLI rests on the model itself; this is a documented, intentional
exception to "caching is what makes a model call deterministic."

---

### Evaluation Framework: Ledger-Driven Metrics, a Thin Runner, and Why It Comes Before LangGraph

**Phase 6 was deliberately run before the LangGraph port (Phase 8), reversing
the originally-documented order.** Evaluation improves research value and
correctness immediately; the LangGraph port only changes execution
plumbing — it does not change pipeline semantics, the ledger schema, or any
metric. Evaluation is therefore robust to LangGraph landing later, and
front-loading it is the higher-value sequencing. This is a recorded,
intentional reorder, not drift — see the Version 1 build order's history
note for the symmetric Retrieval-Integration-before-BM25 precedent.

**Evaluation is, and must remain, a pure, read-only consumer of
`EvidenceLedger`.** Every metric function in `evaluation/metrics/` has the
shape `(EvidenceLedger, gold) -> metric | None` and never mutates the
ledger it's given. This is why `evaluation/` sits at the very bottom of the
dependency chain (`… → rules → orchestration → evaluation`): it may import
from everything and nothing imports from it, and it can score a ledger no
matter what produced it.

**`PipelineRunner` (`evaluation/runner.py`) is evaluation infrastructure,
not production orchestration.** Phase 5 deliberately shipped no
orchestration module, on the understanding that LangGraph would be the
real runner; evaluation cannot score anything without executing the full
pipeline, so Phase 6 forced that question. The resolution: `PipelineRunner`
returns `EvidenceLedger` only, lives entirely inside `evaluation/`, and is
explicitly the seam LangGraph (Phase 8) will replace or absorb —
`EvaluationHarness` and every metric function consume only the ledger it
returns, so neither changes when that happens.

**`EvaluationReport`/`ExampleResult` are deliberately lightweight: fingerprints
and aggregated metrics, never an embedded `EvidenceLedger`.** A report
stores `ledger_fingerprint` (from `evaluation/fingerprint.py`'s
`ledger_fingerprint()`, which excludes `ledger_log`/`reasoning_trace`
timestamps, the same accepted non-reproducibility exception as elsewhere)
and an optional `ledger_path` for on-demand debugging, never the ledger
body itself. `AblationVariant` (defined in `evaluation/config.py`) is
likewise never embedded on `EvaluationReport` — only its name and
`fingerprint()` are stored as plain strings, because embedding it directly
would make `schemas/` (the universal leaf, depending on nothing) depend on
`evaluation/`, inverting the required dependency direction.

**`EvaluationConfig.persist_ledgers` is the single gate for every filesystem
write `EvaluationHarness` performs — both the report and any ledgers.**
`output_dir` only says *where*; `persist_ledgers` says *whether*. This
collapses what would otherwise be two independent, partially-overlapping
flags into one, because `output_dir` is a non-Optional field with a
default — it is always "provided" in the Pydantic sense, so treating its
mere presence as its own trigger would mean every `evaluate_variant()` call
writes to `./evaluation_runs/` by default, including during every test in
this repo. Default (`persist_ledgers=False`): zero filesystem writes,
`ExampleResult.ledger_path` stays `None`. `persist_ledgers=True`: each
example's ledger is written to `output_dir/ledgers/{example_id}_{variant_name}_{ledger_fingerprint}.json`
via `evaluation/reporting.save_ledger()`, and the finished report is
written to `output_dir/{run_id}.json` via the pre-existing `save_report()`.
Per CLAUDE.md's "never silently fail" rule, a config field that exists but
is never read is exactly as forbidden as a swallowed exception — both
fields were dead until this wiring, caught and fixed in the Phase 6 code
review, not the original implementation pass.

**Gold evidence labels are span-based (`GoldSpan`), never chunk-id-based.**
`chunk_id` is a content hash that changes whenever chunking config changes,
so chunk-id gold labels would silently break under re-chunking.
`evaluation/dataset.map_spans_to_chunks()` maps spans to whichever chunks
currently cover them at evaluation time, so the same gold dataset stays
valid across any chunking configuration.

**`FusionStrategy.ROUND_ROBIN` (an evaluation-flavored baseline, never used
by `lncvs.fusion`) and its implementation `round_robin_fuse()` moved from
`evaluation/` to `orchestration/` in Phase 7** — see the "LangGraph
Integration" section below for why. `evaluation/fusion_baselines.py` and
`evaluation/config.py` re-export both for backward compatibility; every
Phase 6 import site (`from lncvs.evaluation import round_robin_fuse`, etc.)
is unchanged. Production fusion remains RRF-only regardless of where the
baseline lives: a boolean "RRF off" toggle is ill-defined once two
retrieval sources exist (BM25 + semantic), since *some* method is still
needed to rank the combined candidate set — round-robin (best-rank-wins)
is that baseline, used solely so the RRF ablation has something concrete to
compare against. It structurally mirrors `lncvs.fusion.rrf.fuse_evidence`'s
grouping and per-claim ranking shape, differing only in the score formula,
and its `rrf_score` reuse on `FusedEvidence` is documented as a generic
ranking score, never an actual RRF score.

**No metric is ever reported as a silent zero when its required gold input
is absent.** `compute_retrieval_metrics` and `compute_citation_metrics`
both return `None` — not `0.0` — when `gold_chunk_ids` is empty or nothing
was cited, respectively; `VerdictMetrics.contradiction_detection_rate` is
`None` when the dataset contains no gold-`CONTRADICTORY` examples. This is
the direct evaluation-layer enforcement of CLAUDE.md's "retrieval metrics
require gold relevance labels; never report partial metrics as complete."

**All verdict metrics are strictly 3-class over `VerdictEnum`.**
`INSUFFICIENT_EVIDENCE` is scored as its own class in accuracy, macro
P/R/F1, and the confusion matrix — never collapsed into a
`CONTRADICTORY`/`CONSISTENT` binary. `datasets/phase6_gold.jsonl` includes
a dedicated `INSUFFICIENT_EVIDENCE` example specifically to exercise this.

---

### LangGraph Integration: Model A, the Reducer Reconciliation, and Two Required Relocations

**Model A (approved): `GraphState` (`schemas/state.py`) is unchanged —
`{ledger: EvidenceLedger, control: ControlState}`.** `EvidenceLedger`
remains the single source of truth, mutated only through `LedgerService`;
the graph adds no second mutation path. The alternative considered and
rejected (Model B) would have hoisted ledger fields up to top-level
LangGraph channels with `add` reducers, shattering the ledger as a single
object and bypassing `LedgerService`'s write-once enforcement — a rewrite
wearing a port's clothes, not a port.

**Reducer reconciliation:** `LedgerService`'s append-only mutation methods
*are* the effective reducer semantics, applied at the `EvidenceLedger`-object
level. At the LangGraph channel level, `orchestration/state_channels.GraphChannels`
declares `ledger` and `control` as whole-object channels with an explicit
`last_write_wins` reducer (`orchestration/reducers.py`) — never the
implicit default. This is valid only because Phase 7 is strictly linear
with no parallel fan-out, so no two nodes ever write to the same channel
within the same super-step. **If a later phase adds parallel fan-out (e.g.
per-claim parallel retrieval), this reducer choice must be revisited.**
`GraphChannels` is LangGraph-facing plumbing local to `orchestration/`,
structurally identical to but never a substitute for `schemas.GraphState`,
which remains the canonical, domain-facing type.

**Nodes call the identical underlying functions `PipelineRunner`
(`evaluation/runner.py`, retained permanently as the equivalence oracle)
calls** — `build_retrieval_queries`, `fuse_evidence`/`round_robin_fuse`,
`classify`, `ThresholdRuleEngine.evaluate`, the same `LedgerService.record_*()`
sequence — in the same order. This is what makes "preserve behavior" a
structural guarantee rather than a hope, verified by
`tests/orchestration/test_graph_equivalence.py`: for every
`standard_ablation_matrix()` variant, `LangGraphPipeline` and
`PipelineRunner` given identical injected models must produce a
byte-identical `ledger_fingerprint`. `ingest_and_index` writes **nothing**
to the ledger, matching the oracle exactly (`PipelineRunner` builds its
indices as local variables before any `LedgerService` call) — this is the
single most important faithfulness constraint in the port.

**Resources never enter `GraphState`.** `PipelineResources` (graph-level:
embedder, NLI model, LLM clients, configs) and `RunContext` (per-run:
variant, the retrievers `ingest_and_index` builds) are plain dataclasses,
not Pydantic models, threaded through LangGraph exclusively via
`config["configurable"]` — never the ledger. A `ChromaIndex` is not domain
state, the same discipline as "the ledger stores chunk IDs, not chunk
bodies" applied one layer up.

**Ablation is handled inside nodes, never via conditional edges.** Each
node reads `run_context.variant` and branches with the identical inline
conditional `PipelineRunner` uses (e.g. `if variant.use_bm25:`). Routing
conditional edges by ablation toggle would create a second place ablation
logic could drift from the oracle; node-internal branching keeps there
being exactly one.

**Error handling:** every node except `error_sink` is wrapped by
`orchestration/nodes._node_error_boundary(stage)`, a decorator that is the
single place a node exception becomes a routed, recorded `StageError` —
appended to `control.errors`, with `control.current_stage` set to `ERROR`
and a conditional edge (`orchestration/nodes.route_after`) sending the run
to `error_sink`, which returns the partial ledger with **no fabricated
verdict**. Per-backend retrieval-failure tolerance (`degraded_sources`) is
deliberately **not** wired: `RetrievalOrchestrator` does not catch
per-backend exceptions today, so there is no existing production behavior
to preserve, and adding that tolerance now would be a new capability —
explicitly out of Phase 7's scope. A claim with zero qualifying NLI
evidence is not an error at all; it flows through to `INSUFFICIENT_EVIDENCE`
exactly as in every other phase.

**No checkpointer.** `build_graph()` compiles with no checkpointer
argument. `LedgerService` mutates `EvidenceLedger` in place, and nodes
return that same object reference as the channel "update" — harmless only
because there is no serialize/restore cycle that could ever observe a stale
copy. Adding a checkpointer later requires reviewing that in-place mutation
first.

**Two relocations Phase 7 required, both disclosed and approved as
deviations during implementation, neither optional:**

1. **`AblationVariant`, `FusionStrategy`, and `standard_ablation_matrix()`
   moved from `evaluation/config.py` to `schemas/`** (`FusionStrategy` →
   `schemas/enums.py`; the rest → `schemas/evaluation.py`).
   `orchestration/nodes.py` must read `variant.use_bm25`/`.fusion_strategy`
   to decide control flow, but `orchestration/` must never import from
   `evaluation/` (the canonical chain is
   `... -> rules -> orchestration -> evaluation`) — both packages instead
   import from the shared leaf. `evaluation/config.py` re-exports all three
   names, so no Phase 6 import site changed.
2. **`round_robin_fuse()` moved from `evaluation/fusion_baselines.py` to
   `orchestration/fusion_baselines.py`**, for the identical reason: the
   `fuse` node and `PipelineRunner` must call the literal same function for
   the equivalence guarantee to hold, and that function must not live
   somewhere `orchestration/` would have to import from `evaluation/`.
   `evaluation/fusion_baselines.py` is now a one-line re-export shim.

Both relocations are why `EvaluationHarness` (`evaluation/service.py`) is
typed against `lncvs.evaluation.runner.LedgerProducer` — a `Protocol`
(`chunking_config` property + `run()` method) — rather than the concrete
`PipelineRunner`. `LangGraphPipeline` satisfies it structurally without
`evaluation/` ever importing `orchestration/`: only call sites (tests, CLI)
import both and wire them together, which is what restores the canonical
`orchestration -> evaluation` dependency edge without `evaluation/`
carrying a hard module-level import of `orchestration/`.

---

## State Management Rules

### GraphState

`GraphState` wraps domain state and control state separately. It is never a flat bag of fields.

```
GraphState = {
  ledger:  EvidenceLedger   # domain state — the single source of truth
  control: ControlState     # orchestration state — not part of the audit trail
}
```

- **`EvidenceLedger`** holds all reasoning state: atomic claims, probe questions, retrieved evidence, fused evidence, NLI results, contradictions, supporting evidence, `unsupported_claims`, reasoning trace, ledger log, and the final verdict.
- **`ControlState`** holds orchestration-only concerns: current stage, errors, retry count, degraded sources, config fingerprint. Control state must never be read by the rule engine — the rule engine consumes only the ledger.
- The ledger stores **chunk IDs**, not full chunk bodies. Chunk text is held in an injected store keyed by ID. Passing full chunk text through LangGraph state at 100k-word scale is a bottleneck and must be avoided.

### Reducer discipline

Declare reducers explicitly for every state field — never rely on default overwrite behavior for list fields.

- Accumulating fields (`atomic_claims`, `retrieved_evidence`, `nli_results`, `reasoning_trace`, `ledger_log`, `errors`) → **append reducers**.
- Scalar fields (`final_verdict`, `current_stage`, `retry_count`) → **last-write-wins**.

### Ledger mutation rules

- The ledger is mutated only through explicit, typed mutation methods — never via direct field assignment from node code.
- The ledger is append-only for audit fields (`ledger_log`); nothing is deleted from it mid-run.
- Every piece of evidence and every NLI result must carry provenance back to a `chunk_id` and character span. If a node cannot supply provenance, it must not write to the ledger.

---

## Logging Standards

Use logging everywhere.

Required levels:

- DEBUG
- INFO
- WARNING
- ERROR

Never use print statements in production code.

---

## Testing Requirements

Every module must include tests.

Minimum coverage target: 80%.

Required tests:

- Unit tests
- Edge cases
- Failure cases
- **Determinism tests** — first-class requirement, not optional. Run the rule engine and the full pipeline N times on identical input and assert identical output, including for any component touching an LLM (via input-hash caching).

Module-specific requirements:

- **Rule Engine**: exhaustive truth-table tests over `{has CONTRADICTED claim, has UNRESOLVED claim, all SUPPORTED}`; threshold-boundary tests; the §14 dummy case must resolve to `CONTRADICTORY`.
- **NLI Verification**: an explicit direction-regression test (premise = evidence, hypothesis = atomic claim) — this is silently reversible and must be pinned by a test, not just code review.
- **Chunking**: chunk-ID stability across repeated runs on identical input; coverage/reconstruction tests.
- **Ledger**: schema validation tests confirming no field accepts a raw `dict`; mutation-method tests; provenance-completeness invariant (every verdict traces to at least one chunk).
- **Retrieval**: contract tests asserting every retriever conforms to the shared interface and returns the typed `RetrievedEvidence` schema; partial-failure isolation (one backend failing does not crash the pipeline).
- **Evaluation**: metric functions tested against hand-computed values (Recall@k, MRR, P/R/F1, contradiction-detection rate).

The dummy test case in `PROJECT_SPEC.md` §14 is the standing end-to-end acceptance test and must be wired into CI starting from the first working vertical slice.

---

## Error Handling

Never silently fail.

Always:

- Raise meaningful exceptions
- Log root causes
- Provide actionable messages

Bad:

```
except:
    pass
```

Good:

```
except Exception as e:
    logger.error(...)
    raise
```

Specific to this system: a retrieval backend failing must not crash the run — it must mark itself `degraded` in `ControlState` and let the remaining backends proceed. Silently treating "no evidence found" as "claim contradicted" is forbidden; that gap must route to `INSUFFICIENT_EVIDENCE` via the rule engine, never be swallowed elsewhere.

---

## Version 1 Scope

Version 1 MUST include:

- Narrative ingestion + text cleaning (offset-preserving)
- Chunking — configurable size/overlap, content-hash IDs, metadata preservation, source traceability
- ChromaDB semantic retrieval (Sentence Transformers)
- BM25 lexical retrieval — added only after the single-retriever slice is proven
- Evidence Fusion (Reciprocal Rank Fusion), with dedup by chunk ID and preserved provenance
- Claim Decomposition Agent
- Question Generation Agent
- NLI Verification Agent (stores label, score, premise, and hypothesis for every result)
- Fully typed Evidence Ledger
- Deterministic Rule Engine producing all three verdicts
- LangGraph workflow (ported in once the linear pipeline is proven correct)
- Evaluation framework (Recall@k, MRR, precision/recall/F1, contradiction-detection rate, citation accuracy, hallucination rate, latency)
- CLI interface
- Determinism via input-hash caching of all LLM/NLI calls

Version 1 MUST NOT include:

- **Knowledge Graph construction** (Version 2)
- **Graph Retrieval** (Version 2)
- Entity & event extraction in support of a graph (Version 2)
- Multi-agent debate
- Fine-tuning or custom transformer training
- Distributed infrastructure
- Autonomous agents
- A web frontend or FastAPI service layer

### Version 1 build order

**As executed (supersedes the original sequence below — recorded here so future sessions don't "fix" it back):**

1. `schemas` + typed `EvidenceLedger` + Rule Engine interface (Phase 0)
2. Single-retriever vertical slice: ingestion → chunking → Chroma → semantic retrieval (Phase 1), then determinism infrastructure — embedding cache, deterministic evidence IDs, config fingerprinting (Phase 1.5)
3. Claim Decomposition (Phase 2a) + Question Generation (Phase 2b), both via a shared `llm/` abstraction with caching
4. **Retrieval Integration** (Phase 3): wire atomic claims + probe questions to retrieval via a claim-agnostic `Retriever`/`Indexer` plus a `RetrievalOrchestrator` that stamps claim/query provenance onto evidence at the ledger boundary
5. BM25 + RRF Fusion (Phase 4) — moved *after* integration, not before: fusion needs claim-linked evidence from multiple sources to be meaningful, so wiring a single source first de-risks adding a second
6. **NLI Verification and Verdict Construction (Phase 5) — complete.** `reasoning/nli/` (evidence-level `NLIVerifier`, cross-encoder model with derived label map, caching decorator) + `rules/classification.py` (pure `classify()` helper) + `rules/threshold_engine.py` (concrete `ThresholdRuleEngine`, the single threshold owner). No new orchestration module — wiring is thin and local to the Phase 5 acceptance test. The PROJECT_SPEC.md §14 dummy case resolves to `CONTRADICTORY`.
7. **Evaluation Framework (Phase 6) — complete.** `evaluation/` (`PipelineRunner`, `EvaluationHarness`, pure metric functions, JSON/matplotlib reporting). See the dedicated "Evaluation Framework" section below for why this was deliberately run *before* the LangGraph port (a documented, intentional reorder of this list — not drift).
8. **LangGraph Integration (Phase 7) — complete.** `orchestration/` (`LangGraphPipeline`, an 8-node `StateGraph` over `GraphState` unchanged, `PipelineResources`/`RunContext` dependency injection, explicit `last_write_wins` channel reducers). See the dedicated "LangGraph Integration" section below for the reducer reconciliation, the `AblationVariant`/`FusionStrategy`/`round_robin_fuse` relocations this phase required, and the cross-variant `ledger_fingerprint` equivalence proof against `PipelineRunner` (retained permanently as the equivalence oracle).
9. Dataset scaling toward 100k+ word narratives

**Original sequence (for history only — do not follow):** decomposition+QG before BM25/fusion was correct; BM25/fusion was originally planned *before* any claim/query integration existed, which would have meant fusing evidence with no claim linkage to fuse *for*. Retrieval Integration was inserted ahead of it once that dependency became clear. Evaluation was originally planned *after* LangGraph; it was promoted ahead of the port once it became clear evaluation improves research value and correctness immediately, while the LangGraph port only changes execution plumbing, not pipeline semantics.

Each phase must end with a runnable, tested system. Do not start the Knowledge Graph, and do not let any module under `graph/` or `extraction/` exist in the Version 1 codebase.

---

## Version 2 Roadmap

Deferred from Version 1, to be added only as measured ablations against the Version 1 baseline:

- Entity & Event Extraction (NER, coreference resolution, entity linking)
- Knowledge Graph construction (NetworkX; typed nodes/edges with provenance)
- Graph Retrieval (entity matching, one-hop/two-hop traversal, temporal neighbor retrieval), integrated into Fusion as an additional `RetrievalSource`

Further future roadmap:

- Temporal reasoning and event-timeline graphs
- Multi-hop graph traversal
- Graded narrative consistency scoring (beyond the ternary verdict)
- Advanced GraphRAG optimization
- Longitudinal character-state tracking

Entry gate for Version 2: do not begin Knowledge Graph work until the Version 1 hybrid (Chroma + BM25) baseline is fully evaluated. Every graph component must demonstrate measurable improvement over that baseline before being adopted, not merely be assumed beneficial.

---

## Evaluation Requirements

Metrics:

- Recall@k, Precision@k, MRR
- Retrieval accuracy
- Citation accuracy
- Hallucination rate
- Answer/verdict faithfulness
- Contradiction detection rate

All experiments must be reproducible: same seed, same config, same data → same metrics.

Retrieval metrics require gold relevance labels per claim; these must exist before Recall@k/MRR can be reported — do not report partial metrics as if they were complete.

---

## Documentation Standards

Every major module requires:

- Purpose
- Inputs
- Outputs
- Limitations
- Future improvements

Document architectural decisions as they are made, not retroactively.

---

## Git Workflow

Commit frequently.

Commit message format:

- `feat:`
- `fix:`
- `refactor:`
- `test:`
- `docs:`

Examples:

```
feat: add semantic chunking pipeline
fix: correct citation mapping bug
refactor: simplify graph traversal logic
test: add rule engine truth-table coverage
```

---

## When Unsure

If multiple implementations are possible:

1. Explain alternatives
2. Compare tradeoffs
3. Recommend one
4. Wait for approval before major architectural changes

Never make significant architectural decisions silently. Never add Knowledge Graph or Graph Retrieval functionality without explicit instruction — it is out of scope for Version 1 by design, not by oversight.

---

## Instructions for Future Claude Code Sessions

When working in this repository, you must:

1. **Treat `PROJECT_SPEC.md` as the source of truth.** If this `CLAUDE.md` and the spec ever appear to diverge, flag the divergence explicitly to the user before proceeding — do not silently resolve it in either direction.

2. **Do not implement, scaffold, or stub Knowledge Graph or Graph Retrieval functionality in Version 1.** No `graph/` module, no `extraction/` module, no NetworkX usage, no graph-shaped fields added to `EvidenceLedger`, even "for later." This includes refusing to add a `RetrievalSource.GRAPH` enum member in Version 1 code.

3. **Never let an LLM produce a final verdict.** If asked to "simplify" the rule engine by having a model decide, or to let an LLM resolve ambiguous cases, push back and explain that this violates a core, explicit project requirement.

4. **Always implement and test for all three verdicts** — `CONSISTENT`, `CONTRADICTORY`, `INSUFFICIENT_EVIDENCE`. If a code path can only ever reach two of the three, that is a bug: it likely means a missing-evidence case is being mislabeled as a contradiction.

5. **Reject untyped data structures at module boundaries.** If you find yourself wanting to pass a `dict` or `List[Any]` into or out of `ledger/`, `retrieval/`, `fusion/`, or `reasoning/`, stop and define a typed model in `schemas/` instead.

6. **Preserve the GraphState split** between `ledger` (domain/audit state) and `control` (orchestration state). Do not let orchestration fields (retry counts, error flags, current stage) leak into the `EvidenceLedger`, and do not let evidence/claims/verdicts leak into `ControlState`.

7. **Do not reorder the Version 1 build sequence.** Do not begin BM25, Question Generation, or LangGraph orchestration before the single-retriever (Chroma-only) vertical slice produces a correct verdict on the `PROJECT_SPEC.md` §14 dummy case.

8. **Before writing code for a new component, restate**: its purpose, inputs, outputs, dependencies, and failure modes — and confirm this matches the architecture in this file and the design review history. If it doesn't, ask before proceeding.

9. **Run the dummy test case mentally (or literally, once code exists) before declaring any retrieval/reasoning component done.** A component that doesn't get the John/London/lost-arm example to `CONTRADICTORY` is not finished, regardless of how clean its code is.

10. **Determinism is testable, not assumed.** Any new component that calls an LLM or embedding model must be paired with caching-by-input-hash and a determinism test, in the same change that introduces it — not as a follow-up.
