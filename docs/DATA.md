# Data access and artifact policy

## MIND-small

The planned external R1 dataset is [MIND-small](https://msnews.github.io/). It is not part of RecForge and is governed by the Microsoft Research License Terms linked from the official dataset page.

As of 2026-09-14, the official page routes MIND-small downloads to the gated Hugging Face dataset `yjw1029/MIND`. Anonymous requests return HTTP 401. Users must independently review and accept the applicable terms and authenticate with Hugging Face; RecForge will not bypass the gate or commit downloaded archives.

Expected upstream artifacts after access is granted:

- `MINDsmall_train.zip`
- `MINDsmall_dev.zip`

Before the adapter is implemented, the project will record retrieval date, source URL, byte size, and SHA-256 checksum. Raw archives, extracted files, processed tables, text features, and checkpoints belong under ignored local artifact directories.

## Synthetic data

`recforge.synthetic.build_synthetic_dataset` generates the only data used by unit tests and the smoke command. It is deliberately small and includes:

- tied popularity counts;
- recent versus old interactions;
- a cold item;
- an item unavailable until after evaluation time;
- multiple user histories and temporal query cutoffs.

Synthetic metrics are correctness checks, not research results.
