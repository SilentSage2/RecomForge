# 0007 — L2 residual low-rank adapter gate

Status: preregistered before adapter training on 2026-09-14.

## Controlled change

Compare the accepted frozen MiniLM ranker against exactly one representation
adaptation: a shared residual bottleneck applied to each frozen 384-dimensional
title feature before the existing layer normalization, 384→64 projection,
masked-mean history aggregation, temperature-scaled dot product, and
impression-local softmax.

The adapter is `x + scale * up(GELU(down(LayerNorm(x))))`, with rank 16, zero-
initialized `up`, and one learned scalar initialized to one. The frozen MiniLM
artifact, split, candidates, negative samples, optimizer, batch size, epochs,
seed, loss, metrics, and evaluation code remain unchanged. This is a low-rank
feature adapter, not LoRA inside the transformer, and must be described as such.

## Gate and stopping rule

Run seed 2027 first. Advance to seeds 2028 and 2029 only if the adapter improves
AUC over the matched frozen-feature seed by at least 0.003 or its 5,000-resample
paired 95% interval remains plausibly positive. Stop after seed 2027 and report a
negative result if the AUC interval is entirely nonpositive. If the point estimate
is between zero and 0.003 with an interval spanning zero, run one confirmation
seed and stop unless both seeds are positive.

The adaptation is accepted only if the three-seed mean AUC effect is positive,
all seeds have the same direction, and added parameter count, peak memory, ranker
runtime, and cached scoring behavior are reported. No rank, activation, dropout,
learning-rate, or epoch search is permitted after inspecting dev labels.

## Interpretation boundary

A positive result would show that a small supervised residual correction improves
the frozen semantic representation under this protocol. A null or negative result
would support the narrower quality–efficiency conclusion for fully frozen
features and blocks further adapter complexity. Neither outcome supports a
MIND-large leaderboard claim.

## Pre-training amendment

The original text initialized both `up` and `scale` to zero. Before any adapter
training, implementation review identified that this is a mathematical dead point:
the residual branch and every branch gradient are zero. Initializing `scale` to
one while retaining a zero `up` matrix preserves the exact identity function at
step zero and permits gradients into `up`. No data, result, or adapter run was
examined before this correction.
