# L1 official-ranking experiment specification

## Research question

Under the official MIND impression-ranking protocol, how much does contextual
title encoding and attentive history aggregation improve ranking over the R1
signed-hash two-tower when candidate sets, training impressions, and evaluation
metrics are fixed?

The falsifiable L1 hypothesis is that an NRMS-style encoder improves AUC by at
least 0.02 on MIND-small dev without using validation labels as features. Failure
to reach that margin is still a reportable result and blocks more expensive
pretrained-text experiments until data or implementation errors are excluded.

## Protocol

- Develop on the released MIND-small train/dev split; repeat the frozen protocol
  on MIND-large only after archive fingerprints and audits are recorded.
- Build the vocabulary from training news only. Unknown and padding tokens are
  distinct, and padding is masked in every attention operation.
- Train from clicked candidates and negatives in the same logged training
  impression. Evaluate against every released candidate in each dev impression.
- Report the official per-impression mean of AUC, multi-positive MRR, nDCG@5,
  and nDCG@10. Do not compare these numbers with temporal-corpus retrieval.
- Select configurations on dev metrics; freeze code and configuration before a
  hidden-test prediction. Public leaderboard feedback is not a tuning split.
- Use seeds 2027, 2028, and 2029 for the final accepted comparison and report
  mean, sample standard deviation, wall time, and peak memory.

## Fixed L1 model

The implementation will be a small typed PyTorch core, not a wrapper around a
recommendation framework:

1. tokenize the title with a deterministic training-only vocabulary;
2. embed at most 30 title tokens;
3. apply one masked multi-head self-attention block;
4. use learned additive attention to pool tokens into a news vector;
5. encode at most 50 prior clicks with one masked multi-head self-attention block;
6. use learned additive attention to pool history into a user vector;
7. score each candidate by user/news dot product.

The initial model uses learned token embeddings and stays below five million
trainable parameters. Frozen or tuned pretrained language encoders belong to L2,
not this baseline.

## Training budget

- Primary loss: impression-local sampled softmax with one clicked item and four
  unclicked items per training example.
- Optimizer: AdamW; one declared learning-rate schedule and early stopping on dev
  AUC with a fixed patience.
- Smoke gate: synthetic fixture and 2,048 MIND-small examples on CPU in under five
  minutes.
- MIND-small gate: at most three configurations and three final seeds; target
  completion within two local-hours per seed.
- MIND-large gate: one frozen configuration, at most eight accelerator-hours per
  seed. Actual device, energy-relevant duration, and monetary cost are logged.

## Baselines and ablations

L1 compares:

- global and time-decayed popularity on logged candidates;
- the existing R1 signed-hash two-tower;
- title mean pooling with the same vocabulary and embedding width;
- the fixed NRMS-style model.

Only two component ablations are required after the full model works: replace
title self-attention with masked mean pooling, and replace history self-attention
with masked mean pooling. Pretrained text, category/entity features, alternative
losses, recency encoding, and ensembles are deferred to L2–L4.

## Acceptance criteria

L1 is complete only when:

1. tokenizer/vocabulary fingerprints and leakage checks are recorded;
2. attention masks, padding invariance, batching, and multi-positive metrics have
   hand-verifiable tests;
3. a clean command trains, evaluates, writes a checkpoint, and emits a strict run
   manifest and official-format prediction file;
4. the three baselines and two required ablations use the same candidate protocol;
5. three final seeds are summarized with uncertainty and failure analysis;
6. README claims distinguish MIND-small dev evidence from any MIND-large hidden-
   test submission.

## First implementation tasks

Implement in this order, merging only after each gate is green:

1. make dev scoring and prediction writing share one deterministic candidate
   ordering implementation; **complete**
2. add the training-only title vocabulary/tokenizer artifact with fingerprints,
   unknown-token handling, truncation, and leakage tests; **complete**
3. add impression-local sampled examples and padded title/history batches;
   **complete**
4. implement masked additive pooling, then the title encoder, user encoder, and
   dot-product scorer as independently tested modules; **complete**
5. run the synthetic and 2,048-example smoke gates before any full training;
   **complete**
6. run the fixed MIND-small baselines, followed by the two declared ablations and
   three-seed comparison.

The `l1_diagnostic_*.json` configurations use the first 10,000 generated training
examples only to catch optimization or implementation failures before expensive
runs. Because this prefix is not a representative sample, diagnostic metrics must
not appear as headline evidence or satisfy the final ablation gate.

The first full-data gate is a paired seed-2027 comparison between
`l1_mind_full_mean_seed2027.json` and
`l1_mind_full_title_attention_seed2027.json`. Both use 64-dimensional
representations, batch size 128, three epochs, all generated training examples,
and the complete dev split. The title-attention variant advances to three seeds
only if it improves full-dev AUC over mean pooling; otherwise mean pooling remains
the efficiency baseline and the failed attention result is reported.
