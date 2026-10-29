# Changelog

Historical package notes are preserved outside the runtime archive in
`docs/status/archive/package-changelog-through-0.6.1.md` and the earlier
`package-changelog-through-0.4.15.md` snapshot in that directory.

## [0.7.0 candidate]

### Removed

Each removed subpath duplicated another export. Import from the replacement:

- `doppler-gpu/run`: import from `doppler-gpu` (same module).
- `doppler-gpu/capsule`: import from `doppler-gpu/host`, which now also exports
  the Capsule identity, migration, signing and release-event helpers and types.
- `doppler-gpu/electron`: import from `doppler-gpu/host/electron` (same module).
- `doppler-gpu/choice-scoring-contract`: import `CHOICE_SCORING_CONTRACT`,
  `snapshotChoiceScoringRequest` and `validateChoiceScoringResult` from
  `doppler-gpu/host` or `doppler-gpu`.

No workspace application or reference example imported these subpaths:

- `doppler-gpu/loaders`: load through `doppler-gpu` or `doppler-gpu/compat`;
  manifest helpers are in `doppler-gpu/tooling/manifest` and OPFS storage in
  `doppler-gpu/tooling/storage`. The explicit loader classes have no replacement.
- `doppler-gpu/orchestration`: no replacement export; Node LoRA loading is
  `loadLoRAFromManifest` in `doppler-gpu/training`.
- `doppler-gpu/serve` (Capsule HTTP adapter), `doppler-gpu/tooling-experimental`,
  `doppler-gpu/diffusion`, `doppler-gpu/energy`: no replacement export.
- `doppler-gpu/models/qwen3`, `models/gemma3`, `models/gemma4`,
  `models/diffusiongemma` and `models/embeddinggemma`: removed with their
  hand-maintained family lists; read model metadata from the shipped
  `models/catalog.json`.

`doppler-gpu/runtime` remains an alias of `doppler-gpu`, and
`createDopplerRuntime` remains an alias of `createDopplerRun`.

### Added

Supported entry points for modules that applications imported by internal path:

- `doppler-gpu/tooling/storage`: `modelExists`, `verifyIntegrity`,
  `loadPersistentModelSource`, `verifyStoredSourceArtifact`, and
  `createBlake3Hasher` (BLAKE3 `createHasher`).
- `doppler-gpu/tooling/device`: `destroyDevice`.
- `doppler-gpu/tooling/manifest`: `DEFAULT_KVCACHE_CONFIG`.
- `doppler-gpu/tooling/runtime`: `initializeInference`.
- `doppler-gpu/tooling`: `runRerank`.
- `doppler-gpu/generation`: `AbortError`, `isAbortError`.
- `doppler-gpu/training` (Node): `loadLoRAFromManifest`.

## [0.6.27 candidate]

- Export resident request/result, tokenization and observation types through
  `doppler-gpu/partitions`; descriptor readiness remains a runtime boolean.
- Add `getRecoveryCapabilities()` with versioned, disabled input replay and
  checkpoint export/import. Live continuation remains bound to resident state.
- Validate required resident methods, descriptor version and recovery claims
  during signed opening. Preserve model qualification and numerical references.
- Installed-package and synthetic contract tests do not qualify physical
  distributed inference or change historical acceptance failures.

## Buffer retirement safety

- Retire only the buffers covered by a queue completion. Buffers retired later
  receive their own completion boundary.
- Confirmed device loss closes the pool without waiting on a destroyed queue.
  Unconfirmed completion or destruction failures retain ownership; calling
  `destroy()` again retries cleanup. Deferred-cleanup statistics retain those
  allocations until destruction succeeds.

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
