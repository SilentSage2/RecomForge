# L3 registered cold-user fallback specification

Status: frozen before candidate-only model training on 2026-09-14.

## Research question

Can a minimal candidate-only propensity model remove the deterministic all-tie
failure for MIND impressions with empty history while leaving every nonempty-
history L2 rank unchanged?

The target population is the 2,214 empty-history impressions identified by the
registered L2 failure slices. This is a cold-user fallback, not a replacement for
personalized ranking.

## Data and model

- Reuse the training-only temporal calibration split: impressions before
  `2019-11-14T00:00:00` fit model weights; the remaining MIND-train impressions
  fit probability calibration only; MIND dev labels are evaluation only.
- Use the immutable 384-dimensional normalized MiniLM item artifact.
- Train exactly one affine scalar head `score = wᵀx + b` with class-balanced
  binary cross entropy, AdamW, learning rate 0.01, weight decay 0.0001, batch size
  8192, three epochs, and seed 2027. The positive weight is computed from the fit
  partition's negative/positive ratio. No hidden layer or feature tuning is
  allowed.
- Precompute one scalar per item. On an empty history only, replace the L2 score
  with the candidate scalar. On every nonempty history, reconstruct the original
  L2 order exactly from its locked prediction file.

Candidate labels are logged clicks and therefore mix semantic propensity with
exposure, publisher, and position bias. The head is not a causal CTR estimator.

## Baselines and metrics

Train global and 72-hour time-decayed positive-click popularity on the identical
fit-model partition. Compare, on the fixed empty-history dev slice:

1. original L2 all-tie scores with stable logged-order resolution;
2. fit-only global popularity with item-ID tie resolution;
3. fit-only time-decayed popularity with item-ID tie resolution;
4. candidate-only affine head.

Report impression count, official AUC/MRR/nDCG@5/nDCG@10, unique-item coverage@1
and coverage@5 over the slice's displayed catalog, and underlying tied-score
impression rate. Use a 5,000-resample paired impression bootstrap for head minus
original tie and head minus the stronger popularity AUC baseline. Also report all
dev official metrics for the original and hybrid predictions, and verify every
nonempty-history rank vector is identical.

## Probability calibration and systems evidence

Fit the already registered positive-scale Platt transform on calibration-tail
candidate scores. Report NLL, Brier, and 15-bin equal-width ECE for raw,
calibrated, and calibration-prevalence constant probabilities on all dev
candidates and on the empty-history slice. Calibration must preserve ranks.

Record parameter count, training time, item-score precomputation time, hybrid
scoring time, peak resident memory, artifact hashes, code state, and dataset
fingerprints.

## Gate and stopping rule

Advance to seeds 2028 and 2029 only if seed 2027:

- improves empty-history AUC over the original tie by at least 0.03 with a paired
  95% interval entirely above zero;
- is no more than 0.005 below the stronger registered popularity AUC baseline;
- causes no negative full-dev AUC effect; and
- preserves every nonempty-history rank.

Accept the fallback only if all three seeds improve empty-history AUC over the
tie, the mean effect is positive, and the full-dev AUC effect is nonnegative in
all seeds. Otherwise retain the result and stop; do not sweep architecture, loss,
cutoff, class weight, learning rate, or epochs.

## Registered qualitative cases

After evaluation, report five empty-history impressions with the largest MRR
regret versus the stronger popularity baseline and five with the largest MRR
gain. Ties are resolved by impression ID. Include candidate count, clicked item
categories, and ranks, but do not present these cases as representative or use
them to alter the model.
