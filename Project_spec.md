# PROJECT_SPEC.md

# Long Narrative Consistency Verification System (LNCVS)

Version: 4.0 (Hybrid GraphRAG Architecture)

---

# 1. Project Vision

Build a system capable of verifying whether a narrative claim is consistent with a long source narrative containing 50,000–100,000+ words.

The system should not rely on a single LLM pass over the entire context.

Instead, it should:

* Retrieve evidence explicitly
* Track reasoning state explicitly
* Verify claims systematically
* Produce deterministic verdicts
* Generate explainable evidence traces

The final system should serve as both:

* A research project investigating long-context reasoning limitations
* A production-style software engineering project demonstrating retrieval, reasoning, graph-based knowledge representation, and agent orchestration

---

# 2. Research Motivation

Recent LLMs support extremely large context windows.

However, larger context windows do not necessarily lead to better reasoning.

Observed failure modes include:

## Long Context Fallacy

Models can access the entire narrative but fail to utilize relevant evidence.

## Attention Dilution

Important facts become buried among thousands of irrelevant tokens.

## Implicit Memory

Facts exist only inside model activations and are not represented explicitly.

## Constraint Loss

Temporal, spatial, causal, and character constraints are not systematically maintained.

## Plausibility Bias

Models often judge whether a claim sounds believable rather than whether evidence supports it.

## Retrieval Failure

Contradictions are frequently missed because contradictory facts are not semantically similar to the queried claim.

---

# 3. Primary Objective

Input:

1. Source Narrative
2. Narrative Claim

Output:

1. Final Verdict
2. Supporting Evidence
3. Contradicting Evidence
4. Evidence Trace
5. Reasoning Summary

Possible Verdicts:

* CONSISTENT
* CONTRADICTORY

The final verdict must be deterministic and reproducible.

---

# 4. Core Hypothesis

Instead of asking a single LLM:

"Is this claim consistent with the narrative?"

The verification process is decomposed into specialized stages:

1. Claim Decomposition
2. Question Generation
3. Hybrid Retrieval
4. Evidence Fusion
5. NLI Verification
6. Evidence Ledger Tracking
7. Deterministic Rule Engine

Every intermediate reasoning step must be observable.

The central hypothesis is that:

Graph Retrieval + Semantic Retrieval + Lexical Retrieval + Explicit Evidence Tracking

will outperform standard RAG systems for long-context consistency verification.

---

# 5. Version 1 Scope

Version 1 MUST include:

✓ Narrative Ingestion

✓ Text Cleaning

✓ Chunking

✓ Entity Extraction

✓ Knowledge Graph Construction

✓ Graph Retrieval

✓ ChromaDB Retrieval

✓ BM25 Retrieval

✓ Claim Decomposition

✓ Question Generation

✓ Evidence Fusion

✓ NLI Verification

✓ Evidence Ledger

✓ Deterministic Rule Engine

✓ LangGraph Workflow

✓ Evaluation Framework

Version 1 MUST NOT include:

✗ Multi-Agent Debate

✗ Fine-Tuning

✗ Custom Transformer Training

✗ Distributed Infrastructure

✗ Autonomous Agents

---

# 6. High-Level Architecture

Narrative
↓
Document Processing
↓
Chunking
↓
Entity & Event Extraction
↓
Knowledge Graph Construction
↓

────────────────────────────────────

Graph Index

ChromaDB Index

BM25 Index

────────────────────────────────────

Claim
↓
Claim Decomposition Agent
↓
Question Generation Agent
↓
Hybrid Retrieval

├── Graph Retrieval

├── ChromaDB Retrieval

└── BM25 Retrieval

↓

Evidence Fusion (RRF)

↓

NLI Verification Agent

↓

Evidence Ledger Update

↓

Deterministic Rule Engine

↓

Final Verdict

---

# 7. Component Responsibilities

## 7.1 Document Processing

Purpose:

Convert raw narrative text into structured chunks.

Responsibilities:

* Load text files
* Clean formatting artifacts
* Preserve chapter metadata
* Preserve source locations
* Generate chunk IDs

Output:

Chunk objects.

---

## 7.2 Chunking

Purpose:

Create retrieval-friendly chunks.

Requirements:

* Configurable chunk size
* Configurable overlap
* Metadata preservation
* Source traceability

Output:

Chunk list with metadata.

---

## 7.3 Entity Extraction

Purpose:

Extract structured entities and events from narrative chunks.

Entity Types:

* Character
* Location
* Event
* Object
* Attribute
* Time Expression

Responsibilities:

* Named Entity Recognition
* Event Extraction
* Attribute Extraction
* Entity Normalization
* Entity Linking

Output:

Structured entity records.

---

## 7.4 Knowledge Graph Construction

Purpose:

Convert extracted entities into a structured graph.

Technology:

* NetworkX

Node Types:

* Character
* Event
* Location
* Object
* Attribute

Relationship Types:

* HAS_ATTRIBUTE
* PARTICIPATED_IN
* LOCATED_AT
* BEFORE
* AFTER
* OWNS
* KNOWS

Output:

Knowledge Graph.

---

## 7.5 Graph Retrieval

Purpose:

Retrieve evidence through graph traversal.

Retrieval Strategy:

* Entity Matching
* One-Hop Traversal
* Two-Hop Traversal
* Temporal Neighbor Retrieval

Output:

Graph-derived evidence.

---

## 7.6 ChromaDB Retrieval

Purpose:

Semantic retrieval.

Technology:

* ChromaDB
* Sentence Transformers

Output:

Semantically relevant evidence.

---

## 7.7 BM25 Retrieval

Purpose:

Lexical retrieval.

Technology:

* rank-bm25

Output:

Keyword-matched evidence.

---

## 7.8 Claim Decomposition Agent

Purpose:

Convert complex claims into atomic facts.

Example:

Input:

John played a two-handed piano piece in London.

Output:

* John played piano
* John used both hands
* Event occurred in London

Goal:

Reduce retrieval ambiguity.

---

## 7.9 Question Generation Agent

Purpose:

Generate retrieval-oriented questions.

Example:

Atomic Claim:

John used both hands.

Questions:

* Did John lose an arm?
* Did John suffer an injury?
* What physical condition did John have?

Goal:

Increase retrieval coverage.

---

## 7.10 Hybrid Retrieval

Purpose:

Combine complementary retrieval mechanisms.

Sources:

1. Graph Retrieval
2. ChromaDB Retrieval
3. BM25 Retrieval

Output:

Candidate evidence set.

---

## 7.11 Evidence Fusion

Purpose:

Merge retrieval outputs.

Method:

Reciprocal Rank Fusion (RRF)

Responsibilities:

* Merge results
* Remove duplicates
* Re-rank evidence
* Preserve provenance

Output:

Unified evidence set.

---

## 7.12 NLI Verification Agent

Purpose:

Verify claims against evidence.

Outputs:

* ENTAILMENT
* CONTRADICTION
* NEUTRAL

Every decision must reference evidence.

---

## 7.13 Evidence Ledger

Purpose:

Maintain explicit reasoning state.

The ledger is the single source of truth.

Responsibilities:

* Track claims
* Track evidence
* Track retrieval history
* Track NLI results
* Track contradictions
* Track reasoning history

---

## 7.14 Deterministic Rule Engine

Purpose:

Generate final verdict.

Hard-Fail Rules:

Rule 1:

Any contradiction
→ CONTRADICTORY

Rule 2:

Missing critical evidence
→ CONTRADICTORY

Rule 3:

All atomic claims supported
→ CONSISTENT

LLMs never generate final verdicts.

Python code generates final verdicts.

---

# 8. LangGraph Design

Required Nodes:

1. Claim Decomposer
2. Question Generator
3. Hybrid Retrieval
4. Evidence Fusion
5. NLI Verification
6. Rule Engine

Shared State:

Evidence Ledger

Every node reads and updates the ledger.

---

# 9. Evidence Ledger Schema

class EvidenceLedger(BaseModel):

```
narrative_chunks: List[dict]

original_claim: str

atomic_claims: List[str]

search_queries: List[str]

graph_evidence: List[dict]

chroma_evidence: List[dict]

bm25_evidence: List[dict]

fused_evidence: List[dict]

nli_results: List[dict]

contradictions: List[dict]

supporting_evidence: List[dict]

reasoning_trace: List[str]

ledger_log: List[str]

final_verdict: str
```

The Evidence Ledger is the central state object used throughout the workflow.

---

# 10. Evaluation Framework

Retrieval Metrics:

* Recall@5
* Recall@10
* MRR

Verification Metrics:

* Accuracy
* Precision
* Recall
* F1 Score

System Metrics:

* Contradiction Detection Rate
* Citation Accuracy
* Hallucination Rate

Performance Metrics:

* Retrieval Latency
* End-to-End Latency

---

# 11. Dataset Strategy

Phase 1:

Synthetic narratives

Phase 2:

Short stories

Phase 3:

Long narratives

Phase 4:

100k+ word narratives

Goal:

Evaluate long-context consistency verification.

---

# 12. Development Philosophy

Build incrementally:

1. Evidence Ledger
2. Entity Extraction
3. Knowledge Graph
4. ChromaDB Retrieval
5. BM25 Retrieval
6. Graph Retrieval
7. Fusion
8. NLI Verification
9. Rule Engine
10. LangGraph Workflow
11. Evaluation Framework

Every stage must be tested before moving forward.

---

# 13. Acceptance Criteria

✓ End-to-end pipeline executes

✓ Graph retrieval operational

✓ Hybrid retrieval operational

✓ Evidence trace generated

✓ Deterministic verdict generated

✓ Evaluation framework operational

✓ Unit tests passing

Minimum Coverage:

80%

---

# 14. Dummy Test Case

Source Narrative:

John lost his left arm in an accident in 2010.

John moved to London in 2012.

Claim:

John played a complex two-handed piano piece at a pub in London.

Expected Verdict:

CONTRADICTORY

Expected Reasoning:

* Claim decomposed correctly
* London evidence retrieved
* Lost arm evidence retrieved
* NLI identifies contradiction
* Rule engine returns CONTRADICTORY

---

# 15. Future Roadmap

* Temporal Reasoning
* Event Timeline Graphs
* Multi-Hop Graph Traversal
* Narrative Consistency Scoring
* Advanced GraphRAG Optimization
* Longitudinal Character State Tracking
