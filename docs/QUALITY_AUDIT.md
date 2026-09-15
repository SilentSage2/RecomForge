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
- The full-data title-attention effect is replicated over three fixed seeds, with
  training-seed sample standard deviations and 5,000-resample paired impression
  bootstraps kept as distinct uncertainty estimates.
- Epoch-boundary checkpoints now atomically preserve optimizer, RNG, progress,
  configuration, and split hashes; interrupted/resumed training has an exact-
  parameter equivalence test.
- Negative and unfavorable results are visible in the README.
- A fixed-revision frozen MiniLM artifact is fully fingerprinted and locally
  reproducible. Its 25,408-parameter ranker improves AUC over matched L1 seeds by
  0.02686 ± 0.00424, with positive paired bootstrap intervals for every seed and
  metric, while reducing mean ranker/evaluation time from 1,349.1 to 63.5 seconds.
- Registered failure slices include counts and within-family multiplicity-adjusted
  intervals. They expose both the cold/dev-only strength and the empty-history,
  6–24-hour exposure-age, and degenerate-publisher limitations.
- The only permitted residual adaptation is retained as a three-seed negative
  result: mean AUC effect −0.00275 ± 0.00519 at 1.55× runtime.
- An auxiliary temporally held-out calibration run uses no dev fitting, preserves
  all ranks, and improves dev NLL/Brier over both raw sigmoid scores and a
  training-prevalence constant. Logged-click bias remains explicit.
- A preregistered 385-parameter cold-user fallback removes the empty-history tie,
  replicates over three seeds, preserves every nonempty rank, and reports the
  accompanying coverage loss, calibration, latency, memory, and fixed failure
  cases rather than presenting AUC alone.

## Evidence gaps

1. **Official-ranking strength:** frozen MiniLM plus a learned projection reaches
   AUC 0.6552 ± 0.0030 on full MIND-small dev and beats matched L1 seeds, but
   remains below the 0.68 aspirational target.
2. **Fair baselines:** global and time-decayed popularity, mean pooling, and the R1
   hash tower now use the official rank-output metrics. The hash result remains a
   smaller, single-seed reference rather than a matched L1 baseline.
3. **Ablation evidence:** title attention versus mean pooling has a full-data,
   three-seed paired comparison. The registered full-data history-attention gate
   is complete and significantly negative on every metric; per its stopping rule,
   it was not expanded beyond seed 2027.
4. **Scale:** complete MIND-small L1 runs exist, but no MIND-large hidden-test
   submission exists.
5. **Uncertainty:** training-seed variance and paired impression uncertainty are
   now reported separately; three seeds are still a modest estimate of training
   variance.
6. **Efficiency beyond the validated path:** cached/batched evaluation is now
   7.97× faster on a locked 2,000-impression comparison with identical metrics;
   its evaluation stage completes all 73,152 dev impressions in 9.14 seconds.
   The L2 path records 1.29–1.32 GB peak resident memory, a 100.2 MB frozen
   feature artifact, and 62.4 seconds of one-time local CPU title encoding.
7. **Research contribution:** the current defensible result is a quality–efficiency
   frontier for frozen semantic representations, with strong cold/dev-only gains,
   a failed supervised adapter, and a minimal cold-user remedy. This is a strong
   research-engineering case study. It is not yet a leaderboard or online-validated
   flagship contribution.

The 10k-example diagnostic found that title attention plus history mean pooling
had the best AUC (0.5425), while full title/history attention reached 0.5335 and
the double-mean baseline reached 0.5330. Because this is one seed on a training
prefix, it is model-selection evidence only. It suggests that history attention
is not justified at low data and defines the two variants that merit full-data
comparison; it does not establish an attention contribution.

The full-data comparison subsequently found an AUC gain of 0.03115 ± 0.00468
across seeds for title attention over double mean pooling. Every seed clears the
predeclared +0.02 gate, and its paired-impression 95% bootstrap interval excludes
zero. The attention model reaches AUC 0.62834 ± 0.00396 versus 0.59719 ± 0.00628,
but costs 6.74× the mean end-to-end CPU time. This supports a stable title-
representation effect under the frozen MIND-small protocol, not a leaderboard or
online-benefit claim.

The next registered experiment isolates pretrained representation quality. Frozen
MiniLM title features plus a shared learned projection reach AUC 0.65521 ±
0.00300, an average paired gain of 0.02686 ± 0.00424 over matched L1 seeds. All
three paired AUC intervals exclude zero, as do MRR and both nDCG intervals. The
result passes the registered L2 gate and supports the representation hypothesis,
but it remains below 0.68 and does not justify a leaderboard claim.

The slice result narrows that conclusion: frozen semantics chiefly help unseen
items and dev-only titles, not all item strata uniformly. A low-rank residual
feature adapter produces one positive seed followed by two significantly negative
seeds, reducing mean AUC while increasing runtime. This blocks adapter escalation
and makes the frozen quality–efficiency result—not parameter-efficient tuning—the
current contribution.

The cold-user experiment then closes the most concrete implementation failure.
The candidate-only head improves empty-history AUC by 0.05384 ± 0.00289 and
full-dev AUC by 0.001252 ± 0.000087, with every nonempty rank unchanged. However,
coverage@1 falls from 18.57% under logged-order tie resolution to 4.18%, and the
labels remain exposure/position biased. The result supports a targeted fallback
and an honest relevance–coverage tradeoff, not general cold-start personalization.

## Release gates

Do not describe RecomForge as flagship-complete or leaderboard-competitive until:

- popularity, mean-pooling, hash-tower, and NRMS-style baselines share candidates,
  splits, and metrics;
- future sequential models are compared against the accepted history-mean result
  under the identical protocol rather than assuming added complexity helps;
- effect sizes, uncertainty, wall time, parameter count, and peak memory are shown;
- a contribution beyond reproduction is isolated by a controlled experiment;
- any result figure satisfies `FIGURE_STANDARDS.md` and is regenerated from locked
  run summaries;
- MIND-large claims are made only after a frozen, accepted hidden-test submission.

Further MIND-small architecture variants are not a substitute for the remaining
external gates: MIND-large hidden-test validation, publisher-time metadata or a
better recency dataset, and online or defensible counterfactual validation of the
cold-user policy.

The smallest acceptable pivot, if NRMS does not learn under the frozen budget, is
to make the negative-source relevance–coverage tradeoff the primary contribution,
strengthen its encoders and objective baselines, and narrow all leaderboard claims.
