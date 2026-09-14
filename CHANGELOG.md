# Changelog

All notable package-facing changes to `doppler-gpu` are documented here. The
complete historical snapshot through 0.4.15 remains intact in
`docs/status/archive/package-changelog-through-0.4.15.md`; it is kept outside
the npm tarball so historical release prose does not consume runtime-package
budget.

## [Unreleased]

### 0.6.2 candidate

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

### Breaking: Capsule naming (0.6.0)

- Replace the Pack product name with Capsule throughout the public API,
  declarations, module paths, configuration, CLI, Forge, and runtime.
  Use `openCapsule()`, `doppler-gpu/capsule`, and `Capsule*` types.
  There are no former-name aliases or old-schema readers in the Capsule runtime.
- Rename identity/receipt fields and signed namespaces. Rebuild and sign new
  Capsules; editing an old signed document is not a valid migration.
- Keep model bytes, tokenizer vocabularies, and retained observations unchanged.
  Historical records are not fresh Capsule qualification.
- See `docs/capsule-naming-migration.md` for consumer and artifact migration.

### Added

- Added strict Doe provider-v1 acquisition with explicit ordered providers,
  typed failures, complete attempt receipts, and restorable global lifecycle.
- Added closed Program Bundles that package hash-verified WGSL and constrained
  host-JS source bytes, plus a generated JSON Schema consumed byte-for-byte by
  Doe.

### Changed

- Program Bundle parity now requires explicit mode and providers and reports
  schema validity, provider availability, execution, and transcript matching
  as independent facts.
- Doppler's Node WebGPU integration delegates to `doe-gpu/node-webgpu` instead
  of maintaining a second provider-signature resolver.
- This contract slice requires the next Doe and Doppler package releases to be
  minor releases; package versions remain unchanged until publish preflight is
  complete.

## [0.4.15] - 2026-07-24

### Added

- Added `generateWithEvidence()` to loaded model handles. The browser-safe
  result binds generated token IDs, transcript, resolved generation config,
  runtime profile, WebGPU backend identity, and execution-plan identity with
  canonical SHA-256 hashes.
