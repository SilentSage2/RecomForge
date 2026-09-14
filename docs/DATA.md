# Data access and artifact policy

## MIND-small

The planned external R1 dataset is [MIND-small](https://msnews.github.io/). It is not part of RecomForge and is governed by the Microsoft Research License Terms linked from the official dataset page.

As of 2026-09-14, the official page routes MIND-small downloads to the gated Hugging Face dataset `yjw1029/MIND`. Anonymous requests return HTTP 401. Users must independently review and accept the applicable terms and authenticate with Hugging Face; RecomForge will not bypass the gate or commit downloaded archives. The maintainer completed authenticated access and verified both archives on that date.

Expected upstream artifacts after access is granted:

- `MINDsmall_train.zip`
- `MINDsmall_dev.zip`

Verified archive fingerprints:

| Artifact | Size shown by Hugging Face | SHA-256 |
|---|---:|---|
| `MINDsmall_train.zip` | 53 MB | `a966e5138ad103376e9817e02395719bf1c62ec56e6e98c30d46fbb991a7fafa` |
| `MINDsmall_dev.zip` | 30.9 MB | `b315cde1c9b9d45008b5a7c4b2e1f87647659f09f74892ae3899c0005d5d6155` |

Raw archives, extracted files, processed tables, text features, and checkpoints belong under ignored local artifact directories. The adapter intentionally keeps the documented MIND clock timezone-naive because the upstream documentation does not declare a timezone; code must not label it UTC without an explicit policy decision.

Reproduce the checked-in temporal protocol audit after extracting both splits:

```bash
recforge-mind-protocol-audit \
  --split train data/raw/MINDsmall_train/behaviors.tsv \
  --split dev data/raw/MINDsmall_dev/behaviors.tsv \
  --output experiments/mind-small/protocol-audit.json
```

The report is a compact derived artifact: it contains only aggregate counts and source-file fingerprints, never user histories or impression rows.

Build the local R1 feature artifact with the accepted primary configuration:

```bash
recforge-mind-features \
  --news data/raw/MINDsmall_train/news.tsv \
  --news data/raw/MINDsmall_dev/news.tsv \
  --dimension 512 \
  --max-history-items 50 \
  --output data/processed/mind-small-content-v1
```

The command refuses to overwrite an existing artifact. Feature semantics and leakage constraints are recorded in ADR 0004.

Build the L1 title vocabulary from training news only. Do not pass the dev or
test news files to this command:

```bash
recforge-mind-vocab \
  --train-news data/raw/MINDsmall_train/news.tsv \
  --max-vocab-size 30000 \
  --min-frequency 2 \
  --max-title-tokens 30 \
  --output data/processed/mind-small-title-vocab-v1
```

The immutable manifest records the training-news fingerprint, configuration,
vocabulary file fingerprint, and semantic vocabulary fingerprint. Dev/test-only
tokens must map to `<unk>`.

## Synthetic data

`recforge.synthetic.build_synthetic_dataset` generates the only data used by unit tests and the smoke command. It is deliberately small and includes:

- tied popularity counts;
- recent versus old interactions;
- a cold item;
- an item unavailable until after evaluation time;
- multiple user histories and temporal query cutoffs.

Synthetic metrics are correctness checks, not research results.
