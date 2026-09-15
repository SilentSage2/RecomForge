# L2 registered failure-slice specification

Status: frozen before computing L1-versus-L2 slice results on 2026-09-14.

## Question and comparison

Where does the frozen MiniLM representation improve or fail relative to the
accepted L1 title-attention model under the identical 73,152-impression MIND-small
dev protocol? The comparison uses candidate-aligned official prediction files
from matched seed 2027. No slice may change candidates, labels, or metric code.

These analyses are diagnostic, not new headline performance estimates. They may
motivate one controlled adaptation test, but no slice is selected or redefined
after observing its effect.

## Registered slice families

Each impression belongs to exactly one slice within each family.

1. **History length:** 0, 1–9, 10–24, 25–49, and at least 50 clicked items.
2. **Candidate-set size:** 2–9, 10–19, 20–49, and at least 50 candidates.
3. **Positive exposure age:** cold (no strictly earlier candidate exposure), at
   most 6 hours, 6–24 hours, 1–3 days, and over 3 days. For multi-positive
   impressions, use the minimum known age. This is a logged-exposure-age proxy,
   not article publication recency. First exposure is built from train candidates
   and then updated online through earlier dev timestamps without dev labels.
4. **Positive item popularity:** cold, tail, middle, and head, using the maximum
   training candidate-exposure count among positive items. Nonzero item cutoffs
   are the training-only 33rd and 67th percentiles with deterministic nearest-rank
   selection.
5. **Candidate-set popularity:** low, middle, and high using the mean
   `log1p(training exposure)` across displayed candidates. Cutoffs are fixed from
   the unlabeled dev candidate sets before prediction metrics are joined.
6. **Positive publisher popularity:** cold, tail, middle, and head, using the
   maximum training candidate-exposure count for a positive item's URL hostname.
   Nonzero publisher cutoffs use training-only 33rd and 67th percentiles.
7. **Positive title rarity:** dev-only, seen/high-rare, and seen/low-rare. A
   positive is dev-only if absent from training news. Otherwise its rare-token
   fraction is the fraction of lowercase Unicode word tokens with training-news
   frequency at most five; high-rare means the maximum positive fraction is at
   least 0.5. Empty titles have rare-token fraction zero.
8. **First positive logged position:** 1–4, 5–9, 10–19, and at least 20, using
   the minimum one-based position among positives. This diagnoses position bias;
   it is not treated as an inference-time feature.

## Metrics and uncertainty

Report sample count, baseline and candidate AUC/MRR/nDCG@5/nDCG@10, paired mean
effect, ordinary 95% paired impression-bootstrap interval, and a within-family
Bonferroni interval for every nonempty slice. Use 5,000 resamples and seed 2027.
Slices with fewer than 200 impressions are reported but marked exploratory.

The report must state how many intervals are examined. Claims are limited to
patterns that retain direction under the within-family interval; isolated nominal
95% intervals are not presented as discoveries. No across-family causal claim is
allowed because the slice attributes are correlated and observational.

## Leakage and artifact requirements

- Training exposures, publisher counts, and token counts use only MIND train.
- Dev labels are used only for official metrics and positive-defined diagnostic
  membership; they do not fit the model, thresholds, or popularity cutoffs.
- Prediction files, behaviors, and news inputs are SHA-256 fingerprinted.
- The evaluator must reject incomplete or misaligned predictions.
- The compact JSON report is committed; raw predictions and datasets remain
  ignored.
