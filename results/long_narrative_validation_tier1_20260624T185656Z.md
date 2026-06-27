# Long-Narrative Validation Report

- Tier: 1
- Started: 2026-06-24T18:55:44.668759+00:00
- Finished: 2026-06-24T18:56:56.303497+00:00
- Chunk count: 1424
- Peak RSS: 809.2 MB
- Model load time: embedder=13.1s, nli=4.8s
- Index build time: chroma=15.7s, bm25=0.1s
- Total NLI truncations: 0
- Determinism match: True
- All sanity checks passed: False
- Any node failure: False

## Verdict distribution

- CONSISTENT: 2
- CONTRADICTORY: 5
- INSUFFICIENT_EVIDENCE: 1

## Per-claim results

| example_id | expected | predicted | sanity_ok | total_latency_s | grounded | citation_accuracy |
|---|---|---|---|---|---|---|
| castaways-consistent-duncan-owner | CONSISTENT | CONSISTENT | True | 10.05 | None | 0.5 |
| castaways-consistent-helena-wife | CONSISTENT | CONSISTENT | True | 2.98 | None | 0.0 |
| castaways-consistent-grant-children | CONSISTENT | CONTRADICTORY | False | 4.85 | None | 0.0 |
| castaways-consistent-paganel-secretary | CONSISTENT | CONTRADICTORY | False | 4.37 | None | 0.0 |
| castaways-contradictory-duncan-owner | CONTRADICTORY | CONTRADICTORY | True | 2.10 | True | 0.5 |
| castaways-contradictory-grant-children | CONTRADICTORY | INSUFFICIENT_EVIDENCE | False | 3.11 | False | 0.0 |
| castaways-insufficient-satellite-phone | INSUFFICIENT_EVIDENCE | CONTRADICTORY | False | 1.71 | None | 0.0 |
| castaways-insufficient-email | INSUFFICIENT_EVIDENCE | CONTRADICTORY | False | 1.76 | None | 0.0 |
