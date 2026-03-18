# Changelog

All notable changes to `@simulatte/doppler` are documented in this file.

This changelog is package-facing and release-oriented. Entries before `0.1.7`
were retrofitted from package version history, release commits, and release
docs so the `0.1.x` line has one conventional npm-visible history surface.

## [0.1.8] - 2026-03-18

### Changed

- Simplified demo to show only verified Q4K models (Gemma 3 270M, Gemma 3 1B).
  Hidden Translate, Diffusion, and Embedding tabs until models are ready.
- Split demo monolith (6,680 lines) into focused modules: core, generation,
  storage, translate, diagnostics, routing, utils.
- Trimmed hosted HF registry and quickstart registry to the two verified models.
- Aligned catalog, HF registry, and quickstart registry to the canonical
  external support registry as single source of truth for HF revisions.
- Renamed all `.mjs` tool scripts to `.js` to match `"type": "module"` convention.
- Switched WebGPU optional dependency from `@simulatte/webgpu` to `webgpu ^0.3.8`.
- Pruned unused `verify:*` npm scripts for models no longer in the active set.
- Updated release-claim policy with newly verified models (LFM2, Qwen 3.5,
  TranslateGemma variants).

### Fixed

- Fixed Qwen 3.5 conversion configs using wrong model preset (`qwen3` instead
  of `qwen3_5`), which caused support matrix check failures.
- Fixed Qwen mRoPE conflation: `ropeInterleaved` was incorrectly set from
  `mropeInterleaved`, forcing adjacent-pair RoPE rotation on Qwen models.
- Fixed catalog lifecycle metadata inconsistencies: corrected `local`, `hf`,
  `curated`, and `demo` fields to match actual artifact availability.
- Fixed GPU-dependent unit tests failing in non-GPU environments by adding
  proper GPU readiness probes with clear skip reasons.
- Fixed kernel-ref digest registry drift (222 vs 224 entries).
- Fixed stale vendor benchmark fixture hashes after compare-engines config update.
- Removed failing and unverified models from demo visibility (TranslateGemma 4B,
  EmbeddingGemma 300M with broken HF manifest, Qwen 3.5 0.8B/2B, F16 variant).

## [0.1.7] - 2026-03-10

### Added

- Added a conventional npm-facing changelog and included it in the published
  package file list.
- Added stronger release-claim, quickstart-registry, local-model-integrity,
  and browser diagnostics regression coverage.
- Added browser OPFS registry smoke workflows for text and embedding model
  validation.

### Changed

- Tightened release-facing model claims around the verified quickstart/catalog
  set and regenerated the support and release matrices from current metadata.
- Synced the public quickstart registry from canonical catalog metadata instead
  of maintaining it by hand.

### Fixed

- Fixed a tensor-loader buffer ownership bug that corrupted returned weight
  buffers and broke Gemma 3 1B generation.
- Fixed quickstart Hugging Face revision drift for registry-backed model IDs.
- Fixed multiple CI contract drifts across onboarding, release metadata,
  support matrices, and generated benchmark fixtures.

## [0.1.6] - 2026-03-07

### Added

- Added stricter config and contract tests around runtime overrides, kernel-path
  semantics, and release-support metadata.
- Added distillation helper extraction coverage for training suite refactors.

### Changed

- Continued the execution-v0 and training orchestration refactor work so public
  entrypoints read more like facades and less like inline policy code.
- Refreshed package exports, repository metadata, and release-facing support
  surfaces for the npm package.

### Fixed

- Preserved explicit `null` semantics for `runtime.inference.kernelPath` through
  schema, runtime config, and harness paths.
- Fixed CI drift around onboarding, registry verification aliases, release
  matrix metadata, and kernel-path preset naming.

## [0.1.5] - 2026-03-06

### Added

- Added diffusion kernel and contract work, plus additional Lean execution
  contract sweep tooling.
- Added public API reference inventory and stronger registry workflow tooling.

### Changed

- Expanded documentation around public APIs, registry workflow, hosted model
  visibility, and release metadata.
- Tightened package exports and release checks for the public package surface.

### Fixed

- Fixed hosted TranslateGemma visibility and registry metadata alignment across
  docs, demos, and package surfaces.
- Removed incorrect self-dependency metadata from the published package.

## [0.1.4] - 2026-03-05

### Added

- Added Lean execution contract scripts and related package commands.
- Added translation prompt validation and quickstart/demo polish.

### Fixed

- Fixed external resolution issues in conversion publication paths.
- Fixed quickstart-facing package and demo issues ahead of publication.

## [0.1.3] - 2026-03-05

### Changed

- Intermediate package metadata and dependency layout refresh during early npm
  packaging work.

## [0.1.2] - 2026-03-05

### Changed

- Aligned build scripts, tests, docs, and package conventions with the active
  workspace and release process.
- Refined README messaging and compatibility notes before npm publication.

## [0.1.1] - 2026-03-05

### Added

- Added benchmark vendor comparison docs, runtime patch documentation, and
  refreshed evidence/chart surfaces for the package release.

### Changed

- Moved vendor benchmark dependencies to development dependencies and kept the
  runtime package dependency surface leaner.
- Refreshed package metadata, exports, and README/API positioning for the first
  public npm publishing cycle.

## [0.1.0] - 2025-12-23

### Added

- Initial npm package release for Doppler.
- Browser and Node command surfaces, CLI entrypoint, loader/storage pipeline,
  RDRR manifest handling, config schemas/presets, WebGPU kernel registry, text
  inference pipeline, conversion tooling, benchmark tooling, tests, and demo
  infrastructure.
