# MIND leaderboard track

## Goal and boundary

RecomForge can target the official MIND News Recommendation Challenge without replacing its primary full-corpus research question. The two tracks share encoders but keep protocols and claims separate:

- `temporal_corpus_v1`: time-eligible whole-catalog retrieval, coverage, tail recall, and latency;
- `mind_official_impression`: ranking only the candidates logged in each official test impression, scored by AUC, MRR, nDCG@5, and nDCG@10.

MIND-small contains training and validation splits only. A real leaderboard submission requires the licensed MIND-large training, validation, and hidden-label test artifacts plus the official prediction format and submission service. Dataset files, predictions, credentials, and checkpoints remain uncommitted.

As checked on 2026-09-14, the [official leaderboard](https://msnews.github.io/) was still receiving 2026 entries. Its displayed leader had AUC 0.7326, while rank 100 displayed AUC 0.7033. These are dated reference points, not acceptance thresholds guaranteed to remain current.

## Staged targets

1. **L0 — Submission correctness.** Vendor no official code; instead test our prediction writer against the official evaluator and sample archive. Reproduce local dev metrics and validate every impression ID, rank permutation, row count, and zip layout.
2. **L1 — Strong transparent baseline.** Implement NRMS-style title self-attention and attentive user pooling, then compare it with the current content-hash towers under the official impression protocol.
3. **L2 — Pretrained text representation.** Compare frozen embeddings, projection-only adaptation, and parameter-efficient encoder tuning. Keep one declared model family and compute envelope rather than sweeping opaque APIs.
4. **L3 — Ranking objectives.** Test pointwise BCE, impression-level sampled softmax, and listwise loss with identical candidates. Add within-impression hard negatives and category/entity features through controlled ablations.
5. **L4 — Sequence and ensemble.** Add recency-aware causal history encoding and, only after single-model evidence, a small seed/model ensemble. Report the marginal gain and inference cost.
6. **L5 — Blind submission.** Freeze code and config before generating test predictions. Record the submission hash and submit sparingly; never tune to public-leaderboard feedback as if it were validation data.

## Success tiers

- **Valid:** an accepted, reproducible official submission.
- **Competitive baseline:** dev and test AUC at least 0.68 with complete ablations.
- **Strong portfolio result:** AUC at least 0.70 plus a clear efficiency or robustness contribution.
- **Leaderboard-frontier attempt:** approach the dated top-100 region while retaining reproducibility and honest compute disclosure.

Ranking position alone is not the portfolio claim. The strongest outcome is a credible submission plus a defensible finding about objectives, representations, temporal robustness, or the relevance–coverage tradeoff.

## Immediate prerequisites

- confirm authenticated access to all three MIND-large archives and record their fingerprints;
- add an official-format prediction writer and validator;
- reproduce the official evaluator byte-for-byte on a tiny fixture and MIND-small dev;
- freeze an L1 experiment spec and hardware budget before adding the NRMS-style model.
