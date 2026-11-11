# Changelog

Historical package notes are preserved outside the runtime archive in
`docs/status/archive/package-changelog-through-0.6.1.md` and the earlier
`package-changelog-through-0.4.15.md` snapshot in that directory.

## [0.7.1] - 2026-11-11

Patch release. Published bytes for 0.7.0 are unchanged.

### Fixed

- **Gate activations agree across fused and unfused paths.** 0.7.0 removed
  the +/-15 sigmoid input clamp from `silu.wgsl`, but
  `sigmoid_gated_add_f32.wgsl` and the `moe_shared_slot_*` kernels kept it.
  For |gate| > 15 the separate and fused paths disagreed, and the Qwen
  shared-expert gate differed from the logistic function: `sigmoid(-20)`
  gave 3.06e-7 instead of 2.06e-9. All of them now use `silu.wgsl`'s
  unclamped form; only the explicit `swigluLimit` output clamp remains. The
  two Qwen 3.6 35B-A3B q4_0 runtime profiles carry the current kernel digests.
- **Repeated prompts no longer exhaust an explicit device memory budget.**
  With `configureDeviceMemoryBudget`, idle pooled buffers count as live.
  Each new prompt length added buckets on top of earlier ones until a later
  request was rejected. A new generation's first prefill now retires idle
  pooled buffers after confirmed GPU completion, only when a budget is set.
  Owned buffers are never retired. Measured with Qwen 3.5 2B under a
  6,442,450,944-byte budget: steady memory between requests stays at or
  below 2.6 GB.

### Added

- `model.inspect.generate(prompt, { capture })` narrows activity recording by
  kind, layer, token range and budget (selective capture, `e56cf5d6`).

### Known limits

- Budgeted hosts should also set
  `inference.session.prefillChunkSubmitMode: "sync"` (or
  `prefillMaxInFlightChunks: 1`) for models whose manifest declares async
  prefill submits. With async submits, warm requests keep up to three layer
  chunks of prompt-sized temporaries alive. Qwen 3.5 2B at 2,000 tokens
  peaked at 4.40 GB with sync, against a failure under the same 6.44 GB
  budget with async, at the same prefill time.
- Flash-Next (Qwen 3.8) is not qualified. Its 2,187-token prompt still fails
  the full-logit gate against independent CPU references; see
  `reports/flash-next-reference-controls-20261110/`.

## [0.7.0] - 2026-11-10

First npm release since 0.6.1. It also carries the 0.6.27 and 0.6.2 candidate
notes below, which were never published separately.

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

### Changed

- MoE softmax top-k for F32 router logits now uses `softmax_topk_ranked.wgsl`:
  one invocation per expert ranks the probabilities in parallel. The maximum,
  exponent sum and top-k renormalization keep the scalar kernel's arithmetic
  order, so indices and weights are bitwise identical (ties order by lower
  expert index).
- The MoE router projection honors the kernel path's `moe_router` step for the
  active phase; it previously ignored the kernel path.
- Native GGUF shared-slot MoE adds an opt-in `pair_f32` variant selected by the
  kernel-path step: aligned two-block word loads, staged inputs, subgroup
  reductions, a 2D (row group, route) dispatch and a fixed slot-order combine.
  Stored GGUF bytes are unchanged.

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
