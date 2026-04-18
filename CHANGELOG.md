# Changelog

All notable changes to `doppler-gpu` are documented in this file.

This changelog is package-facing and release-oriented. Entries before `0.1.7`
were retrofitted from package version history, release commits, and release
docs so the `0.1.x` line has one conventional npm-visible history surface.

## [Unreleased]

## [0.4.3] - 2026-04-18

### Added

- `"sideEffects"` entry in `package.json` restricts module-load side effects
  to WGSL files and `src/gpu/device.js`. Bundlers can now tree-shake unused
  exports from every other module.
- Narrow tree-shakable subpath exports split the `doppler-gpu/tooling`
  mega-barrel:
  - `doppler-gpu/tooling/device` — GPU init, capability probing, shader
    preseed helpers.
  - `doppler-gpu/tooling/storage` — OPFS shard manager, registry, inventory,
    quota, cache.
  - `doppler-gpu/tooling/manifest` — RDRR manifest parsing + schema
    defaults.
  The existing `doppler-gpu/tooling` barrel stays for back-compat.
- `doppler-gpu/structured` subpath export for JSON grammar-mask and
  structured-generation helpers (`createJsonGrammarMask`).
- `doppler-gpu/client/model-manager` subpath for the runtime model manager
  (`initDoppler`, `loadModel`, `getPipeline`, `getCurrentModelId`,
  `unloadModel`).
- `doppler-gpu/provider` now re-exports `wrapPipelineAsHandle` alongside
  `createDopplerProvider` so consumers reach both via one import.
- Per-family static metadata modules:
  - `doppler-gpu/models/qwen3`
  - `doppler-gpu/models/gemma3`
  - `doppler-gpu/models/gemma4`
  - `doppler-gpu/models/embeddinggemma`
  Each exports `FAMILY_ID`, `HF_REPO_ID`, `KNOWN_MODELS` (frozen list of
  `{modelId, label, sourceModel, hfPath, defaultRuntimeProfile, modes}`),
  plus `resolveModel(modelId)` and `resolveHfBaseUrl(modelId, revision)`.
  Kilobyte-scale, tree-shakable, no runtime weight.
- `registerShaderSources(map)` and `hasPreseededShaderSource(name)` on
  `doppler-gpu/tooling/device`. Consumers that bundle WGSL via
  `import.meta.glob('.../kernels/*.wgsl', { as: 'raw', eager: true })`
  preseed the shader cache to bypass the runtime HTTP-fetch path entirely.
- `docs/library-consumers.md` — migration + usage guide for applications
  embedding `doppler-gpu` as an npm dependency.
- `ensureModelCached` exposed through the tooling surface
  (`doppler-gpu/tooling` and `doppler-gpu/tooling/storage`).

### Fixed

- Removed top-level `await` from `src/rules/rule-registry.js`. The 46 rule
  JSON files are now imported statically with `with { type: 'json' }` import
  attributes, which is synchronous at module load. This unblocks iife worker
  bundles (e.g., Vite/Rollup classic-format workers) that previously failed
  to bundle `doppler-gpu`.

### Changed

- Swapped Qwen 3.5 0.8B Q4K kernel refs for the 6 full-attention layers
  (indices 3/7/11/15/19/23): `attn_stream` → `attn_head256`, and
  `q4_prefill` `multicol_shared` → `q4_widetile` (register-tiled). Manifest
  and `src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json` regenerated.
  Strix Halo / RDNA-3 warm-cache 3-run means showed prefill@80 +57.5%
  (σ≈3.8), prefill@15 +16.6%, decode flat (σ≈1.3); `doppler verify` match
  with the expected Qwen `<think>` wrapper. Receipts live on the Strix
  Halo workspace; `models/catalog.json` carries the canonical summary.
- Retuned `profiles/qwen-3-5-0-8b-throughput` for AMD Strix Halo / RADV Mesa 26.
  Switched from `batchSize=8 readbackInterval=8` to `batchSize=4
  readbackInterval=2` with batch-level stop-check. On Strix Halo this raises
  Qwen 3.5 0.8B Q4K decode from ~32.0 to ~34.2 tok/s (+7%), drops TTFT from
  ~390 ms to ~155 ms (-60%), and lifts prefill from ~41 to ~104 tok/s (+150%).
  Cross-checked against `gemma-3-1b` to confirm the win is Qwen-specific
  (linear-attention + larger vocab) and doesn't belong in the global default
  profile. Receipts under `reports/qwen-3-5-0-8b-q4k-ehaf16/`.

### Added

- `--manifest-only` and `--bootstrap` flags on
  `tools/publish-hf-registry-model.js`. `--manifest-only` skips shard
  staging for republish of an existing model when only the manifest
  changed (e.g., kernel-ref swap). `--bootstrap` is required for the
  first publish of a new model (entry with `lifecycle.availability.hf:
  false`, `hf.{repoId,path}` set, `hf.revision` absent) and flips
  `availability.hf` to `true` in the local catalog after a successful
  upload. Both paths now write the newly minted `hf.revision` back into
  `models/catalog.json` so the local catalog is not drifted from the
  hosted `registry/catalog.json`.
- Documented the perf-validated kernel-path promotion workflow in
  `docs/style/benchmark-style-guide.md`. Two-stage flow: runtime override +
  correctness/perf gates first; promotion to the conversion config and
  manifest regeneration only after both gates pass on the target hardware.
- `tools/probes/qwen-decode-sweep.sh` for repeating the Strix Halo sweep when
  re-tuning the model-scoped profile.

## [0.2.2] - 2026-04-13

### Fixed

- Switched the README benchmark image to an absolute GitHub raw URL so the npm
  package page renders the current published chart instead of relying on an
  unpublished local asset path.

## [0.2.1] - 2026-04-13

### Changed

- Bumped the package version past the already-published `0.2.0` release.
- Pointed the README demo CTA at the canonical public demo URL:
  `https://d4da.com/doppler`.
- Clarified the README benchmark image scope so the hero chart stays tied to the
  published MacBook Air M3 warm-cache lanes instead of implying whole-catalog
  coverage.
- Expanded the README verified-model note to include Gemma 4 E2B alongside the
  other verified local-artifact models outside the quickstart registry.
- Refreshed release-facing metadata so the release matrix now surfaces the
  Gemma 4 E2B compare lane and current package version.

### Fixed

- Synced the public quickstart registry revisions with the canonical catalog for
  `gemma-3-1b-it-q4k-ehf16-af32` and `qwen-3-5-0-8b-q4k-ehaf16`.

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
- Switched WebGPU optional dependency to `webgpu@^0.4.0`.
- Pruned unused `verify:*` npm scripts for models no longer in the active set.
- Updated release-claim policy with newly verified models (LFM2, Qwen 3.5,
  TranslateGemma variants).

### Fixed

- Fixed Qwen 3.5 conversion configs using the wrong family identifier (`qwen3`
  instead of `qwen3_5`), which caused support matrix check failures.
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
  matrix metadata, and kernel-path config naming.

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
  RDRR manifest handling, config schemas/assets, WebGPU kernel registry, text
  inference pipeline, conversion tooling, benchmark tooling, tests, and demo
  infrastructure.
