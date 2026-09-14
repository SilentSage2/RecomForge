# ADR 0005: Uniform-negative comparison policy

- **Status:** Accepted for R1
- **Date:** 2026-09-14

## Decision

Compare in-batch negatives with a shared uniform pool while holding training queries, batch size, epochs, model, optimizer, features, seed, and evaluation protocol fixed.

For each query in a batch, sample one item uniformly from its time-eligible catalog after excluding its history and all current positives. The resulting batch-sized pool is shared across users. A per-user mask removes pool items that are future, previously consumed, or positive for that user. Every row therefore has at least its own verified legal negative, while the item tower encodes only one additional batch-sized pool rather than hundreds of independent negatives per user.

## Rationale

- Sampling respects the same temporal catalog used in evaluation.
- The shared pool provides roughly batch-sized competition without multiplying item-tower compute by the batch size.
- Validity masks prevent false negatives caused by another user's sampled item.
- The comparison isolates negative-source policy more cleanly than changing model capacity or evaluation candidates.

## Limitation

The uniform pool requires one additional batch of item encodings, whereas in-batch training reuses positive encodings. Runtime and trained-pair counts must therefore be reported alongside quality. This is a controlled practical comparison, not a claim of exactly matched FLOPs.
