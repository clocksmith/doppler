# Training Hash Policy

Canonical hash policy for training/UL artifacts.

## Hash function

- Use `sha256Hex()` from `src/utils/sha256.js`.
- Do not introduce alternate digest algorithms for training provenance paths.

## Content hash vs file hash

1. `manifestHash` / `manifestContentHash`
- Deterministic hash over normalized manifest content view.
- Excludes volatile fields from the hashed view.

2. `manifestFileHash`
- Hash of serialized file bytes on disk.
- May differ from content hash if formatting/volatile fields differ.

## Stage linkage

1. Stage2 dependency accepts explicit `stage1ArtifactHash`.
2. Stage2 verifies:
- stage1 file hash / manifest hash compatibility
- stage value (`stage1_joint`)
- contract hash (`ulContractHash`)
- latent dataset hash/shape integrity

## Policy requirement

- All new training artifact linkage fields must use SHA-256 hex and be validated in `tools/verify-training-provenance.mjs`.
