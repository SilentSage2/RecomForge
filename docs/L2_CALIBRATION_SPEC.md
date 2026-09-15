# L2 training-only calibration specification

Status: frozen before calibration split creation or calibration fitting on
2026-09-14.

## Registered result

The split contains 126,695 fit-model and 30,270 calibration impressions. On all
2,740,998 dev candidate exposures, the fitted scale is 0.232693 and bias is
−5.713429. Calibration reduces NLL from 9.93818 to 0.16830 and Brier score from
0.93785 to 0.03874, while preserving every impression's rank permutation. It also
slightly improves on the calibration-prevalence constant (NLL 0.17003, Brier
0.03899). Its ECE is 0.00280 versus 0.00233 for the constant baseline, so no ECE
superiority is claimed. The registered NLL/Brier gate passes.

## Question

Can the accepted frozen MiniLM ranker's logged-candidate scores be converted into
better click-probability estimates without using MIND dev labels or changing
candidate ranking?

This is an auxiliary calibration experiment. The headline L2 model trained on all
MIND train remains the ranking result; it cannot be retrospectively calibrated on
interactions it already used for fitting.

## Temporal partition and model

- Fit-model partition: MIND-small train impressions strictly before
  `2019-11-14T00:00:00` in the dataset's timezone-naive local clock.
- Calibration partition: train impressions at or after that cutoff.
- Evaluation partition: all MIND-small dev impressions.
- Retrain the frozen MiniLM projection ranker from seed 2027 on the fit-model
  partition with the accepted architecture, optimizer, negative sampling, three
  epochs, and feature artifact. The calibration labels are never used to update
  ranker weights.

The source file is not chronological, so the split command parses every row's
timestamp independently and preserves relative source order within each output.
It records timestamp inversions and fingerprints the source and both generated
files. Split files remain ignored.

## Calibrator and metrics

Fit a two-parameter monotonic Platt transform
`p(click) = sigmoid(positive_scale * score + bias)` on every logged candidate in
the calibration partition. Parameterize `positive_scale` with softplus so rank
order is exactly preserved; fit with deterministic full-batch LBFGS. No parameter
is selected on dev.

Report candidate-weighted binary negative log likelihood, Brier score, and
15-bin equal-width expected calibration error on calibration and dev. Include the
raw `sigmoid(score)` and a constant probability equal to calibration prevalence as
references. Also verify that calibrated and raw official ranks are identical for
every dev impression.

These probabilities estimate logged click labels and inherit position/exposure
bias. They are not causal click-through rates or online calibration evidence.

## Acceptance and stopping

Calibration is useful if dev NLL and Brier both improve over raw sigmoid while
rank permutations remain identical. ECE is reported but is not a sole acceptance
criterion because it depends on binning. A failure is retained and stops further
calibrator searches; no bin-count, cutoff, optimizer, or functional-form sweep is
permitted after dev evaluation.
