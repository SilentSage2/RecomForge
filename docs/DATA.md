# Data access and artifact policy

## MIND-small

The planned external R1 dataset is [MIND-small](https://msnews.github.io/). It is not part of RecForge and is governed by the Microsoft Research License Terms linked from the official dataset page.

As of 2026-09-14, the official page routes MIND-small downloads to the gated Hugging Face dataset `yjw1029/MIND`. Anonymous requests return HTTP 401. Users must independently review and accept the applicable terms and authenticate with Hugging Face; RecForge will not bypass the gate or commit downloaded archives. The maintainer completed authenticated access and verified both archives on that date.

Expected upstream artifacts after access is granted:

- `MINDsmall_train.zip`
- `MINDsmall_dev.zip`

Verified archive fingerprints:

| Artifact | Size shown by Hugging Face | SHA-256 |
|---|---:|---|
| `MINDsmall_train.zip` | 53 MB | `a966e5138ad103376e9817e02395719bf1c62ec56e6e98c30d46fbb991a7fafa` |
| `MINDsmall_dev.zip` | 30.9 MB | `b315cde1c9b9d45008b5a7c4b2e1f87647659f09f74892ae3899c0005d5d6155` |

Raw archives, extracted files, processed tables, text features, and checkpoints belong under ignored local artifact directories. The adapter intentionally keeps the documented MIND clock timezone-naive because the upstream documentation does not declare a timezone; code must not label it UTC without an explicit policy decision.

## Synthetic data

`recforge.synthetic.build_synthetic_dataset` generates the only data used by unit tests and the smoke command. It is deliberately small and includes:

- tied popularity counts;
- recent versus old interactions;
- a cold item;
- an item unavailable until after evaluation time;
- multiple user histories and temporal query cutoffs.

Synthetic metrics are correctness checks, not research results.
