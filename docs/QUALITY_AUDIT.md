# Substantive quality audit

Audit date: 2026-09-14. This document distinguishes a sound research foundation
from a flagship result. Passing tests and having complete documentation are
necessary engineering evidence, not substitutes for scientific contribution.

## Current evidence

The central question is important and falsifiable: under leakage-safe temporal
retrieval, how do candidate learning and negative sources trade relevance against
catalog coverage and tail exposure? The three-seed R1 comparison gives one real,
bounded finding: uniform shared negatives improve Recall@100 by 0.0432–0.0506 in
every paired seed but produce zero tail recall, whereas in-batch clicked negatives
retain broad coverage and nonzero tail recall. This refutes the preregistered idea
that in-batch negatives are a universal relevance improvement.

That finding is worth retaining, but its scope is narrow. The title representation
is a signed hash, the availability timestamp is reconstructed, and neither learned
retriever is yet a strong modern recommendation baseline. The result supports a
claim about objective behavior in this implementation and protocol; it does not
support state-of-the-art relevance, production benefit, or leaderboard quality.

## Current strengths

- The official logged-impression protocol and reconstructed whole-corpus temporal
  retrieval protocol are explicitly separated.
- Split, metric, candidate, feature, and run artifacts are fingerprinted; critical
  edge cases and leakage boundaries have tests.
- R1 uses three fixed seeds and reports sample standard deviation rather than
  presenting a favorable single run.
- The L1 path implements title and history self-attention, masked additive pooling,
  impression-local negatives, official metrics, and deterministic prediction
  ordering directly in PyTorch. It is not an API wrapper or mock.
- Negative and unfavorable results are visible in the README.

## Evidence gaps

1. **Official-ranking strength:** the L1 NRMS-style model has only a 1,024-example
   smoke run (AUC 0.5313). It has not beaten a fair baseline.
2. **Fair baselines:** logged-candidate popularity, title mean pooling, and the R1
   hash tower must be evaluated through the identical official metric path.
3. **Ablation evidence:** title self-attention and history self-attention have not
   been individually removed under a fixed budget.
4. **Scale:** no complete MIND-small L1 training run or MIND-large hidden-test
   submission exists.
5. **Uncertainty:** L1 has no three-seed result or paired per-impression bootstrap.
6. **Efficiency beyond the validated path:** cached/batched evaluation is now
   7.97× faster on a locked 2,000-impression comparison with identical metrics;
   its evaluation stage completes all 73,152 dev impressions in 9.14 seconds.
   Peak memory remains unmeasured.
7. **Research contribution:** NRMS reproduction alone is baseline engineering, not
   novelty. A flagship contribution still needs a defensible result about negative
   objectives, recency/sequence modeling, pretrained representation efficiency,
   or relevance–coverage-aware reranking.

## Release gates

Do not describe RecomForge as flagship-complete or leaderboard-competitive until:

- the cached evaluator's peak memory is recorded alongside its existing full-dev
  wall time;
- popularity, mean-pooling, hash-tower, and NRMS-style baselines share candidates,
  splits, and metrics;
- the fixed NRMS-style model completes MIND-small and beats a meaningful baseline,
  or the failure is diagnosed and reported as a bounded negative result;
- the two registered attention ablations and three final seeds are complete;
- effect sizes, uncertainty, wall time, parameter count, and peak memory are shown;
- a contribution beyond reproduction is isolated by a controlled experiment;
- any result figure satisfies `FIGURE_STANDARDS.md` and is regenerated from locked
  run summaries;
- MIND-large claims are made only after a frozen, accepted hidden-test submission.

The smallest acceptable pivot, if NRMS does not learn under the frozen budget, is
to make the negative-source relevance–coverage tradeoff the primary contribution,
strengthen its encoders and objective baselines, and narrow all leaderboard claims.
