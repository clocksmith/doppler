# Changelog

All notable package-facing changes to `doppler-gpu` are documented here. The
complete historical snapshot through 0.4.15 remains intact in
`docs/status/archive/package-changelog-through-0.4.15.md`; it is kept outside
the npm tarball so historical release prose does not consume runtime-package
budget.

## [Unreleased]

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
