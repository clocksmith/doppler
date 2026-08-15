# CATSCAN: Artifact Storage

Component: `doppler.runtime-source.storage`

Parent: [Shipped Source](../CATSCAN.md)

## Target

Persist, retrieve, verify, and transport artifact bytes through explicit storage contexts and integrity contracts.

## Authority

- Owns storage backends, artifact contexts, shard retrieval, downloads, exports, quotas, and persisted inventory.
- Does not own artifact interpretation, inference behavior, or silent model substitution.

## Scope

- OPFS, IndexedDB, memory, file, and HTTP artifact storage mechanisms.

## Contracts

- Input: Artifact locations, validated identities, shard metadata, integrity policy, and backend configuration.
- Output: Verified byte ranges and lifecycle-bound [artifact storage contexts](artifact-storage-context.js).

## Invariants

- Storage and GPU execution remain orthogonal.
- Retrieved bytes preserve declared artifact identity and integrity status.
- Missing, stale, or mismatched artifacts fail visibly.

## Acceptance

- Backend, shard, download, integrity, close, and range-read tests pass.
- Evidence: [storage tests](../../tests/storage).

## Non-goals

- Parsing model semantics or deciding which artifact an application should trust.

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.
