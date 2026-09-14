# R0–R1 experiment specification

## Hypothesis

Under a strict temporal protocol, a compact two-tower retriever using recent user history and item metadata will improve target-item Recall@K over non-personalized popularity baselines. In-batch negatives should improve retrieval over uniformly sampled negatives, but may concentrate recommendations on popular items.

## Primary comparisons

1. Global popularity versus time-decayed popularity.
2. Best popularity baseline versus compact two-tower retrieval.
3. Uniform sampled negatives versus in-batch negatives with model, examples, optimizer budget, and evaluation candidates fixed.

## Protocols

- Logged-impression ranking reports AUC, MRR, NDCG@5, and NDCG@10.
- Temporal corpus retrieval reports Recall@20/100, MRR@20, coverage@100, head/mid/tail recall, and exact-search latency.
- User state contains only events strictly earlier than the query timestamp.
- Candidate eligibility and item-availability policy are versioned before the first external run.

## Compute envelope

- Tests and smoke evaluation run on CPU.
- The first meaningful learned run fits one accessible GPU.
- Development uses one seed; the final comparison targets three seeds.

## Acceptance criteria

R0 is complete when schemas, fixtures, metrics, temporal validators, two popularity baselines, CI, and the synthetic smoke command are green.

R1 is complete when MIND access and fingerprinting are documented, one data command and one experiment command are reproducible, the two-tower comparison and negative-sampling ablation are complete, and actual results plus failures are reported in the README.

## Non-goals

ANN search, online serving, feature stores, distributed training, deep ranking, sequence transformers, LLM reranking, and UI work are deferred until R1 passes.
