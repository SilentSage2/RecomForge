# Cold-user result figure contract

- Claim: a 385-parameter candidate-only fallback removes the empty-history all-tie
  failure consistently across three seeds without changing any nonempty-history
  rank, while its full-dev effect remains necessarily small.
- Inputs: `experiments/mind-small/l3-cold-user-three-seed-aggregate.json` and the
  seed-level run IDs and prediction hashes named there.
- Population: all 2,214 MIND-small dev empty-history impressions; no subsampling.
- Panels: empty-history AUC for original tie, fit-only global popularity, fit-only
  72-hour time-decayed popularity, and the candidate head; plus candidate-head
  AUC effect on empty-history and full-dev populations.
- Uncertainty: mean ± sample standard deviation over seeds for head metrics;
  seed-level 5,000-resample paired intervals for head-minus-tie effects.
- Axes: AUC in absolute units and AUC difference in percentage-point units; no
  truncated axis without an explicit break marker.
- Missing runs: fail rather than omit a declared seed.
- Outputs when plotted: `figures/generated/l3-cold-user.png` and `.pdf`; generated
  files remain ignored until visual QA.
- Caption boundary: logged-click propensity is exposure/position biased and the
  result is MIND-small dev evidence, not online cold-start performance.
