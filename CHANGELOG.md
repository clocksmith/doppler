# Changelog

All notable package-facing changes to `doppler-gpu` are documented here. The
complete historical snapshot through 0.4.15 remains intact in
`docs/status/archive/package-changelog-through-0.4.15.md`; it is kept outside
the npm tarball so historical release prose does not consume runtime-package
budget.

## [Unreleased]

## [0.6.0] - Release candidate

### Pack operations and application integration

- Expose `openPack().executeOperation()` for generation, embeddings, reranking,
  and sequence encoding with validated inputs, bounded output, cancellation,
  and completion receipts binding the exact request and executed Pack.
- Add signed embedding qualification against pinned CPU source vectors and
  an explicit F16 Qwen embedding conversion recipe. Retain the rejected Q4
  numerical result separately; it is not a qualified embedding configuration.
- Return explicit null GPU timing fields when those measurements are unavailable,
  keeping real reranking observations valid under the strict JSON contract.
- Require application-owned persisted release checkpoints for Pack v3 history.
  Qualification receipts and independent adoption retain separate authority.

The physical Reploid document experiment and its exact local 0.5.2 candidate
tarballs are retained at `clocksmith/reploid` revision `ca25da6`, under
`docs/status/document-search-2026-09-06`. They do not qualify this 0.6.0 candidate
until its own installation and execution checks pass. Publication and independent
operator reproduction remain separate release evidence.

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
- This contract slice requires compatible minor releases of Doe and Doppler.
  The Node provider contract needs its own published-provider verification;
  browser qualification does not establish that dependency's availability.

## [0.4.15] - 2026-07-24

### Added

- Added `generateWithEvidence()` to loaded model handles. The browser-safe
  result binds generated token IDs, transcript, resolved generation config,
  runtime profile, WebGPU backend identity, and execution-plan identity with
  canonical SHA-256 hashes.
