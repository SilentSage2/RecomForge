# RecomForge instructions

RecomForge is a research-grade recommendation project, not a tutorial or product demo.

- Keep every headline claim tied to a versioned experiment and leakage-safe protocol.
- Implement and test evaluation, temporal splits, and baselines before neural models.
- Keep logged-impression ranking distinct from corpus retrieval.
- Prefer typed, dependency-light Python and explicit interfaces.
- Do not commit datasets, checkpoints, secrets, or raw run artifacts.
- Record seeds, data fingerprints, Git revision, environment, hardware, duration, and metrics for every research run.
- Add controlled ablations; change one research variable at a time.
- Run formatting, linting, typing, unit tests, and the CPU smoke command before merging.
- Do not add ANN search, serving, distributed training, or LLM components before the R0–R1 acceptance criteria are met.
