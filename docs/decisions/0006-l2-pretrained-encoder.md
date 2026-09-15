# 0006 — L2 frozen pretrained title encoder

Status: accepted for the L2 MVP on 2026-09-14.

## Decision

Use `sentence-transformers/all-MiniLM-L6-v2` at immutable Hugging Face revision
`826711e54e001c83835913827a843d8dd0a1def9` for the first frozen title artifact.
The repository declares Apache-2.0 at that revision. Encode at most 32 wordpiece
tokens, apply attention-mask-aware mean pooling to the final hidden states, and L2
normalize each 384-dimensional float32 vector.

Use Hugging Face Transformers only for checkpoint/tokenizer loading and the base
forward pass. Pooling, normalization, artifact validation, projection, history
aggregation, training, and evaluation remain explicit RecomForge code. Set
`trust_remote_code=False` and prefer safetensors.

## Rationale

- Six transformer layers and roughly 91 MB of model weights make a local CPU/MPS
  preflight practical.
- The 384-dimensional representation is materially richer than L1 while remaining
  cheap enough to cache once for all 65k MIND-small news items.
- A frozen artifact isolates representation quality from end-to-end tuning and
  directly tests the bottleneck suggested by L1.
- An immutable revision and explicit pooling rule avoid silent changes from a
  mutable model alias or library default.

## Alternatives deferred

- A larger BERT/DeBERTa encoder is deferred until the small frozen encoder proves
  the representation hypothesis.
- SentenceTransformers' high-level `encode` wrapper is not used in the artifact
  implementation because pooling and normalization must be inspectable.
- LoRA/QLoRA and full fine-tuning are deferred until the frozen gate is evaluated.
- Sparse, multi-vector, and cross-encoder models belong to later retrieval and
  reranking experiments rather than this controlled L2 comparison.

## Required artifact metadata

Record model ID and revision, tokenizer settings, pooling, normalization, dtype,
embedding dimension, item/title hashes, dependency versions, device, batch size,
duration, throughput, peak resident memory, artifact byte count, and SHA-256.
