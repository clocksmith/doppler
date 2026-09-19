# Changelog

Historical package notes are preserved outside the runtime archive in
`docs/status/archive/package-changelog-through-0.6.1.md` and the earlier
`package-changelog-through-0.4.15.md` snapshot in that directory.

## [0.6.2 candidate]

- Add explicitly selected operation stream v2 with stable Unicode text additions,
  incremental token and embedding events, hash chaining, final reconstruction
  verification and public accumulators. Operation v1 remains available unchanged.
- Preserve shared device resources when another session binds the same physical
  GPU; retain cleanup and device-generation regression coverage.
- Test one installed archive through standalone capabilities and the reconciled
  Reploid library provider, including cancellation and request-bound adapters.
- Version 0.6.2 identifies new candidate bytes. Published 0.6.1 and earlier
  unpublished 0.6.1 candidates remain distinct historical artifacts. This entry
  is not publication or physical qualification evidence for the new archive.
- Migration: explicitly send `doppler.capsule-operation-request/v2`, consume
  additions with `createCapsuleStreamAccumulator()`, and require `finish()` before
  accepting completion. Calling `snapshot()` after every event deliberately
  recreates cumulative copying. Capsule/model identities do not change merely
  because an application adopts a different transport format.

## Capsule naming migration (0.6.0)

Use `openCapsule()`, `doppler-gpu/capsule`, and `Capsule*` types. Former Pack names
and schemas have no runtime aliases. Rebuild and sign migrated Capsules;
editing signed documents is invalid. See `docs/capsule-naming-migration.md`.
