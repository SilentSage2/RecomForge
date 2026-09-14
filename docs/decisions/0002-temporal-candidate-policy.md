# ADR 0002: Temporal candidate-universe policy

- **Status:** Accepted for R1
- **Date:** 2026-09-14

## Context

MIND supplies impression timestamps, ordered history IDs, and news metadata, but it does not supply a reliable article publication timestamp. Full-corpus retrieval still requires a time-eligible item universe; using every article in the released files would place future items into earlier queries and create an unrealistic evaluation.

## Decision

Define an item's `first_observed_at` as the earliest impression timestamp at which the item appears either:

1. in the displayed candidate set; or
2. in the user's pre-impression click history.

At query time `t`, the corpus contains items with `first_observed_at <= t`. By default, previously consumed history items are removed, except that current positive targets are retained if they are repeats.

The index is built from train and development behavior files without consulting click labels. Using all released behavior files is transductive availability reconstruction, not model training: future-only items remain ineligible for earlier queries. The exact input file fingerprints are part of the experiment manifest.

## Rationale

- Candidate appearance proves the item was available by that impression.
- History appearance proves the item was available before that impression, although the exact earlier time is unknown.
- The rule is deterministic, label-independent, and prevents future-only catalog leakage.
- Keeping repeated positive targets prevents the seen-item filter from making a valid query impossible.

## Limitations

- `first_observed_at` is an upper bound on true availability, not publication time.
- An item may have been available well before its first released observation.
- History items lack individual timestamps, so all we know is that they predate the containing impression.
- Full-corpus unlabeled items are not verified negatives and may include relevant unclicked content.
- Results are specific to this reconstructed corpus and must not be compared directly with logged-impression leaderboard metrics.

## Reporting requirement

Every result table must include a `protocol` column. Use `logged_impression` for exposed-candidate ranking and `temporal_corpus_v1` for this reconstructed retrieval corpus. Candidate-universe size, seen-item filtering, and first-observation inputs must be reported alongside retrieval metrics.
