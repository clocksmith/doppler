# Deterministic Timestamp Policy (Training)

Timestamp behavior for reproducible training/UL artifacts.

## Policy

1. Timestamps are recorded for auditability:
- directory names
- `createdAt`
- report `timestamp`

2. Deterministic hashes must not depend on volatile timestamps:
- `manifestHash` is computed from a deterministic manifest view that excludes volatile timestamp/run-id fields.

3. File hashes remain byte-level strict:
- `manifestFileHash` reflects exact serialized bytes and may vary when timestamp fields vary.

## Required checks

1. Deterministic content hash stability across timestamp changes.
2. Stage-link hash checks use deterministic and/or file hash according to contract.
3. Repro docs must clearly separate content hash and file hash semantics.
