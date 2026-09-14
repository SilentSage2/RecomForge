# Run bundles

Every experiment writes a new directory under `runs/`. The directory name combines a UTC timestamp, experiment name, and random suffix so concurrent launches do not collide. Existing directories are never overwritten.

```text
runs/<run-id>/
  manifest.json
  metrics.json
  model.pt        # learned experiments only
```

The manifest records:

- schema version and run status;
- start/end timestamps and measured duration;
- full command, resolved configuration, and seed;
- Git commit and dirty-worktree state;
- dataset fingerprints;
- Python version and non-identifying platform/machine information;
- SHA-256 of the canonical metrics artifact.

Learned-run metrics also record the SHA-256 of `model.pt`, making the checkpoint traceable without committing it.

`runs/` is ignored because full experiment output can become large. Selected compact manifests and aggregate results may be copied into `experiments/` after review. A run from a dirty worktree is valid for development but must not support a headline result.

Generate a local example with:

```bash
recforge-smoke
```
