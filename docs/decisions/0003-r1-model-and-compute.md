# ADR 0003: R1 model boundary and compute envelope

- **Status:** Accepted for R1
- **Date:** 2026-09-14

## Decision

R1 uses a compact PyTorch two-tower retriever. Independent user and item MLP towers consume deterministic feature vectors, emit L2-normalized embeddings, and train with temperature-scaled in-batch softmax. Exact dot-product search remains the evaluation reference.

Feature construction stays outside the neural module so the negative-sampling comparison can hold model architecture and inputs fixed. The first external run will use item text/category features and a strictly pre-query aggregation of recent-history item features; it will not use learned IDs for unseen development items.

The development reference machine is an Apple M4 laptop with 24 GB unified memory. Unit tests and tiny smoke runs must pass on CPU. Local research runs may use PyTorch MPS, while reported runs must record device, duration, peak memory where available, seed, and package versions. The first full run must fit this machine or one commodity GPU; distributed training is out of scope.

## Rationale

- A small explicit model makes the negative-sampling experiment interpretable.
- Normalized embeddings keep exact cosine-style retrieval simple and stable.
- Content-derived features provide a defensible path for cold items.
- A feature/model boundary lets tests exercise the loss before the MIND featurizer exists.
- PyTorch supports both CPU tests and the local MPS backend without a second framework.

## Deferred choices

The text featurizer, history length, embedding width, temperature, and symmetric-loss flag will be fixed in a checked-in experiment config before training. ANN search, sequence encoders, hard-negative mining, and ranking remain deferred until R1 acceptance criteria pass.
