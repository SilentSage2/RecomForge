# ADR 0004: R1 content and history features

- **Status:** Accepted for R1
- **Date:** 2026-09-14

## Decision

The first learned MIND experiment uses a 512-dimensional signed feature hash over lowercased title tokens plus namespaced category and subcategory values. The hash is fixed and does not learn a vocabulary from train or development data. Abstract text is an explicit optional ablation and is off in the primary run.

The user vector input is the normalized mean of content features for the most recent 50 items in the released pre-impression history. Empty histories map to an all-zero vector. No current-impression candidate or label contributes to the user feature.

Feature artifacts contain sorted item IDs, a float32 matrix, source hashes, configuration, and output hashes. Local matrices remain ignored; only compact configs, manifests, and result summaries may be committed.

## Rationale

- Corpus-independent hashing avoids vocabulary fitting across the temporal boundary.
- Content features provide representations for development items unseen as training targets.
- A mean history encoder is intentionally weaker and easier to audit than a sequence model.
- Fixed-dimensional dense vectors keep the first two-tower experiment small enough for exact retrieval and local execution.

## Limitations

Hash collisions lose information, word order is discarded, and mean pooling ignores recency within the selected history. These limitations make the model a controlled learned baseline; learned text encoders and causal sequence models remain later milestones.
