# L2 pretrained title-representation experiment specification

## Motivation and research question

L1 isolates a stable title-representation effect: contextual title attention
improves AUC by 0.03115 ± 0.00468 over title mean pooling, while replacing history
mean pooling with history self-attention significantly reduces every ranking
metric. The next falsifiable question is therefore:

> Under the identical leakage-safe MIND impression protocol, how much ranking
> quality can a frozen pretrained text representation add over the accepted L1
> title encoder, and what encoding, memory, storage, and serving cost does that
> gain require?

The initial hypothesis is that frozen pretrained title embeddings improve
MIND-small dev AUC by at least 0.015 over the L1 three-seed mean of 0.62834. AUC
0.68 remains an aspirational competitive target, not an acceptance threshold or a
claim. A failed gate is reported and blocks parameter-efficient tuning until data,
pooling, and projection errors are excluded.

## Registered result

The frozen MVP passes its gate. Across seeds 2027–2029 it reaches AUC 0.65521 ±
0.00300, MRR 0.30752 ± 0.00485, nDCG@5 0.33716 ± 0.00453, and nDCG@10 0.40049 ±
0.00438. The paired AUC gain over the matched accepted L1 seeds is 0.02686 ±
0.00424; every per-seed 5,000-resample paired interval excludes zero. The model
has 25,408 trainable parameters and takes 63.5 ± 2.4 local CPU seconds for three
training epochs plus full-dev evaluation. One-time title encoding takes 62.4
seconds and stores 100.2 MB.

This supports the frozen representation hypothesis and permits one controlled
adaptation experiment. It does not meet the aspirational 0.68 target or establish
MIND-large leaderboard performance. The locked result is in
`experiments/mind-small/l2-frozen-three-seed-aggregate.json`.

The permitted rank-16 residual feature adapter does not replicate. Its AUC effect
relative to the matched frozen model is +0.00240, −0.00266, and −0.00798 across
seeds, for a mean of −0.00275 ± 0.00519. It adds 13,457 parameters and increases
mean ranker/evaluation time by 1.55×. The registered stopping rule therefore
rejects the adapter and blocks further representation-adaptation searches.

The registered seed-2027 failure slices localize the frozen model's improvement
to cold-exposure and dev-only positive titles. Training-seen low-rarity titles do
not retain a gain under the within-family interval, the 6–24-hour exposure-age
proxy is negative, and empty histories remain unsolved. Full results and
multiplicity rules are in `docs/L2_FAILURE_SLICE_SPEC.md` and
`experiments/mind-small/l2-frozen-seed2027-failure-slices.json`.

The design follows prior MIND evidence that pretrained language representations
can improve news encoders while making repeated news encoding a central systems
cost:

- MIND dataset paper: <https://aclanthology.org/2020.acl-main.331.pdf>
- PLM-in-the-loop news recommendation: <https://arxiv.org/abs/2102.09268>
- efficient PLM news recommendation: <https://aclanthology.org/2022.emnlp-main.368.pdf>

## Smallest meaningful MVP

1. Use `sentence-transformers/all-MiniLM-L6-v2` at revision
   `826711e54e001c83835913827a843d8dd0a1def9` under Apache-2.0, with at most 32
   wordpiece tokens, explicit masked-mean pooling, and L2 normalization. The full
   rationale is in `docs/decisions/0006-l2-pretrained-encoder.md`.
2. Encode every unique MIND-small train/dev title exactly once. Store a
   fingerprinted float32 artifact containing item IDs, embeddings, source-title
   hashes, model revision, tokenizer settings, device, precision, and duration.
3. Freeze those embeddings. Train only a learned projection to 64 dimensions and
   the accepted masked-mean history aggregator with candidate dot products.
4. Use the existing impression-local 1-positive/4-negative training examples,
   official full-dev candidate sets, stable rank tie policy, and official
   AUC/MRR/nDCG implementation without modification.
5. Run seed 2027 as a gate. Advance to seeds 2028 and 2029 only if AUC improves by
   at least 0.015 over the matched L1 seed or the paired interval remains
   plausibly positive.

This MVP tests pretrained representation quality rather than a new user model. It
does not require end-to-end transformer fine-tuning, a cross-encoder, an LLM API,
or a recommendation framework.

## Frozen comparisons

All rows use the same MIND-small train/dev split and logged candidates:

1. global and 72-hour time-decayed popularity;
2. title mean + history mean;
3. accepted L1 title attention + history mean;
4. frozen pretrained title embedding + learned projection + history mean.

Only after row 4 passes its gate may L2 add one parameter-efficient tuning
variant. That variant must preserve the base checkpoint revision and compare
LoRA/QLoRA-style trainable parameter count, peak accelerator memory, wall time,
and title-encoding throughput against the frozen row. Full fine-tuning is not the
default.

## Evaluation and uncertainty

Primary metric: official impression-level AUC. Secondary metrics: official MRR,
nDCG@5, and nDCG@10. Report:

- three fixed training seeds for any accepted headline model;
- sample mean and sample standard deviation across seeds;
- 5,000-resample paired impression bootstrap against accepted L1 predictions;
- parameter count, peak host and accelerator memory, end-to-end training time,
  one-time title encoding time, artifact bytes, and warm-batch encoding throughput;
- failure slices by history length, candidate popularity, unseen-in-train title
  token rate, and candidate category, with sample counts and uncertainty;
- inference latency after title embeddings are cached separately from uncached
  transformer encoding latency.

Calibration is deferred until a training-only calibration partition is frozen.
No temperature or threshold may be selected on the released dev labels and then
reported on those same labels as unbiased calibration evidence.

## Data and leakage boundaries

- Training labels may fit the projection/ranker; dev labels are evaluation only.
- The text encoder may consume train and dev title text because item content is an
  inference-time feature, but it may not train on MIND dev interactions or labels.
- Model selection on MIND-small dev remains development evidence. MIND-large
  hidden test is required for an official leaderboard claim.
- The embedding artifact must fail validation if item order, title content, model
  revision, tokenizer settings, dimensionality, dtype, or row count changes.
- No dataset, pretrained weight, title embedding artifact, token, or run checkpoint
  is committed to Git.

## Remote compute preflight

Before accelerator work, record without exposing credentials:

- SSH host alias and project path, but never private-key contents;
- GPU model/count, driver, CUDA runtime, PyTorch CUDA version, supported dtypes,
  available RAM, persistent disk, and quota or monetary cost;
- repository commit and clean status;
- dataset and vocabulary fingerprints matching the accepted local protocol;
- a synthetic forward/backward smoke test and a 256-title encoding benchmark;
- atomic checkpoint/resume behavior on the remote filesystem.

The first L2 artifact and all registered runs intentionally execute locally. A
remote preflight is deferred until a later experiment actually requires more
compute; remote access is not a dependency of the frozen MVP.

## Acceptance and stopping rules

The frozen-embedding MVP is complete only when the artifact contract, leakage
tests, synthetic test, full seed-2027 run, paired bootstrap, and cost measurements
are complete. Advance to three seeds at the registered gate above. Advance to
parameter-efficient tuning only when the frozen encoder provides a defensible
quality gain or a clearly diagnosed representation limitation makes tuning the
minimal next test.

Do not describe L2 as leaderboard-competitive without a frozen MIND-large hidden-
test submission. Do not continue scaling if quality fails while cost rises unless
a controlled error analysis identifies a falsifiable remedy.
