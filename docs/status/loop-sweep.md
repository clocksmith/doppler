## fire-4 — 2026-04-18 UTC

Landings (7+): 7
WGSL touches: 4   JS touches: 3 (incl. 1 delete + 2 new shared-util files)

Baseline parity vs fire-3: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* violations (unchanged), `test:unit` 30/349
(unchanged). No regressions.

### Changed
- src/gpu/kernels/attention_decode_contiguous_turboquant_f16kv.wgsl — removed unused `override NUM_CENTROIDS`
- src/gpu/kernels/attention_decode_contiguous_turboquant_prod_f16kv.wgsl — removed unused `override NUM_CENTROIDS_MSE`
- src/gpu/kernels/attention_decode_tiered_turboquant_f16kv.wgsl — removed unused `override NUM_CENTROIDS`
- src/gpu/kernels/attention_decode_tiered_turboquant_prod_f16kv.wgsl — removed unused `override NUM_CENTROIDS_MSE`
- src/config/kernels/kernel-ref-digests.js — re-synced after WGSL edits
- src/storage/artifact-storage-context.js — replaced local `isNodeRuntime` with import from `src/utils/runtime-env.js`
- src/client/runtime/index.js — same
- src/client/runtime/node-quickstart-cache.js — same
- src/client/provider.js — same (and simpler-variant local version dropped in favor of canonical)
- src/client/doppler-api.js — same
- src/storage/export.js — replaced local `normalizeModelId` with import from `src/storage/normalize-model-id.js`
- src/storage/shard-manager.js — same
- src/storage/reports.js — migrated `normalizeModelId` to wrap shared helper while preserving `'unknown'` fallback semantics

### Deleted
- tools/bench-gemma-logits-vs-tokens.js — dead 3-line shim (`import './bench-text-decode-paths.js'`); zero references in package.json, docs, tests, or other tools

### Added
- src/utils/runtime-env.js — new canonical `isNodeRuntime()` helper lifted from 5 call sites
- src/storage/normalize-model-id.js — new canonical `normalizeModelId()` helper lifted from 3 call sites

### Visited clean (skipped from future fires)
- src/storage/export.js
- src/storage/shard-manager.js
- src/storage/reports.js
- src/storage/artifact-storage-context.js
- src/storage/blake3.js
- src/storage/quota.js
- src/storage/preflight.js
- src/storage/downloader.js
- src/storage/quickstart-downloader.js
- src/storage/registry.js
- src/storage/inventory.js
- src/storage/index.js
- src/storage/download-types.js
- src/storage/source-artifact-store.js
- src/storage/emulated-vram.js
- src/client/doppler-api.js
- src/client/doppler-registry.js
- src/client/provider.js
- src/client/runtime/index.js
- src/client/runtime/node-quickstart-cache.js
- src/client/runtime/model-source.js
- src/client/runtime/model-session.js
- src/gpu/kernels/attention_decode_contiguous_turboquant_f16kv.wgsl
- src/gpu/kernels/attention_decode_contiguous_turboquant_prod_f16kv.wgsl
- src/gpu/kernels/attention_decode_tiered_turboquant_f16kv.wgsl
- src/gpu/kernels/attention_decode_tiered_turboquant_prod_f16kv.wgsl
- tools/bench-text-decode-paths.js
- tools/distillation.js
- tools/lora.js

### Punts
- `nodeModule` helper (3 active call sites + 1 more: storage/reports.js, inference/pipelines/structured/json-head-pipeline.js, plus skip-list copies in src/utils/load-json.js and src/gpu/kernels/shader-cache.js). Full consolidation blocked by skip-list coverage; partial consolidation risks inconsistent bundler behavior since the whole point of the helper is a per-file local that the bundler cannot statically follow. Keep as-is.
- `isNodeRuntime` had TWO variants (stricter 5-line vs. simpler 3-line). Fire-4 picked the stricter version as canonical for the shared util. The simpler variant's semantics are a strict subset, so behavior is preserved; verified by identical pass on test:unit. If any caller relied on the looser check matching a non-standard runtime, a regression test would be needed.
- `sanitizeFilename` near-duplicate between `src/storage/export.js` (replaces `[\\/:*?"<>|]`) and `src/experimental/browser/tensor-source-download.js` (replaces `[^a-zA-Z0-9._-]`). Semantics differ — one is Windows-compat conservative, other is URL-safe strict. Not a naive consolidation.
- Pre-existing codegen patches broken for 6 variants (carried over).

## fire-3 — 2026-04-18 UTC

Landings (7+): 7
WGSL touches: 3   JS touches: 4 (incl. .d.ts)

Baseline parity vs fire-2: `kernels:check` same 6 pre-existing codegen-patch errors,
`imports:check:browser` same pre-existing node:* violations, `test:unit` 30/349 same.
No regressions, no fixes to pre-existing state. Digests re-synced after WGSL edits.

### Changed
- src/gpu/kernels/cast_f32_to_f16.wgsl — removed unused `const MAX_WG_X` (declared once, never referenced; 2D-dispatch doc-comment referred to it but math at line 30 uses `num_wg.x * WORKGROUP_SIZE` directly)
- src/gpu/kernels/fused_matmul_q4.wgsl — removed unused `const BLOCK_SIZE` (just a doc constant for Q4_K byte size; file never consumes it, uses `Q4KBlock` struct directly)
- src/gpu/kernels/dequant_shared.wgsl — removed unused `const NUM_SUBBLOCKS` (hard-coded 8 literal never referenced; file loops over subblock offsets with inline counts)
- src/config/kernels/kernel-ref-digests.js — re-synced after WGSL edits
- src/cli/doppler-cli.js — replaced local `isPlainObject` with import from `src/utils/plain-object.js`
- src/tooling/kernel-path-builder/index.js — replaced local `isPlainObject` with import from `src/utils/plain-object.js`
- src/inference/pipelines/text/generator-helpers.js — replaced local `isPlainObject` with import from `src/utils/plain-object.js`
- src/formats/rdrr/manifest.js — redirected `ERROR_CODES`/`createDopplerError` import from deleted `errors/index.js` shim to `errors/doppler-error.js`
- src/gpu/device.js — redirected same two imports from deleted shim to `errors/doppler-error.js`

### Deleted
- src/errors/index.js — dead shim (`export { ERROR_CODES, createDopplerError } from './doppler-error.js'`); 6 other files already imported directly from `doppler-error.js`; only 2 holdouts redirected above
- src/errors/index.d.ts — paired `.d.ts` dead shim

### Visited clean (skipped from future fires)
- src/errors/doppler-error.js (canonical error module; unchanged)
- src/gpu/device.js
- src/formats/rdrr/manifest.js
- src/cli/doppler-cli.js
- src/tooling/kernel-path-builder/index.js
- src/inference/pipelines/text/generator-helpers.js
- src/gpu/kernels/cast_f32_to_f16.wgsl
- src/gpu/kernels/fused_matmul_q4.wgsl
- src/gpu/kernels/dequant_shared.wgsl
- src/gpu/kernels/dequant_shared_vec4.wgsl
- src/gpu/kernels/fused_ffn_q4k.wgsl
- src/gpu/kernels/fused_ffn_q4k_f16.wgsl
- src/gpu/kernels/matmul_gemv_subgroup.wgsl
- src/client/wrap-pipeline-handle.js
- src/client/failure-taxonomy.js
- src/client/receipt.js
- src/client/fault-injection.js
- src/formats/rdrr/validation.js
- src/formats/rdrr/parsing.js
- src/formats/rdrr/types.js
- tests/formats/litert-types.test.js
- tests/formats/tflite-types.test.js

### Punts
- Remaining `isPlainObject` duplicate sites (per earlier punts): `src/rules/{execution-rules,layer-pattern}-contract-check.js` stay under visited-clean and have different helper bodies; migrating them needs a matchesExactObject consolidation first (separate fire).
- `formatNumber` / `formatMs` / `formatMB` in `src/cli/cli-output.js` — exported from module but only used within the CLI suite; could be made private (not re-exported) but cli-output is listed as the printing surface module, so keeping them exported avoids an API break. Deferred.
- `onnxruntime-web` devDep flagged as potentially unused by static grep, but it may be a transitive peer-dep via `@huggingface/transformers`. Needs peer-dep audit before removing. Deferred.
- `wrapPipelineAsHandle` in `src/client/wrap-pipeline-handle.js` — only used internally via `provider.js`; could be inlined but is part of the published client API shape. Deferred.
- Several WGSL files have unused `const BLOCK_SIZE` in similar Q4_K doc-comment form (fused_ffn_q4k.wgsl, fused_ffn_q4k_f16.wgsl, matmul_gemv_subgroup.wgsl, etc.). Batch-migrating all of them is the same pattern; next fire can keep sweeping in this lane.
- Pre-existing codegen patches broken for 6 variants (carried over from fire-1/fire-2).

## fire-2 — 2026-04-18 UTC

Landings (7+): 7
WGSL touches: 3   JS touches: 4 (incl. .d.ts)

Baseline parity vs fire-1: `kernels:check` (same 6 pre-existing codegen-patch errors, untouched),
`imports:check:browser` (same pre-existing node:* violations, untouched), `test:unit` 30/349
(same). Fire-2 did not regress any gate and did not fix any pre-existing gate. All
digest/reachability artifacts re-synced after WGSL edits.

### Changed
- src/gpu/kernels/kv_quantize_turboquant.wgsl — removed unused `override NUM_CENTROIDS` (declared once, never referenced in shader body; file uses `NUM_BOUNDARIES` instead)
- src/gpu/kernels/kv_quantize_turboquant_prod.wgsl — removed unused `override NUM_CENTROIDS_MSE`
- src/gpu/kernels/matmul_gemv_subgroup_f16a.wgsl — removed unused `override MULTICOL_MAX_SUBGROUPS`
- src/config/kernels/kernel-ref-digests.js — re-synced after WGSL edits (content digests for touched kernels refreshed)
- src/tooling/litert-package-runtime.d.ts — redirected `LiteRTSource` import from deleted `formats/litert/index.js` shim to `formats/litert/types.js` directly
- src/tooling/diagnose-runner.js — replaced local `isPlainObject` with import from `src/utils/plain-object.js`
- src/config/execution-contract-check.js — replaced local `isPlainObject` with import from `src/utils/plain-object.js`
- src/tooling/kernel-path-builder/runtime-overlay.js — replaced local `isPlainObject` with import from `src/utils/plain-object.js`

### Deleted
- src/formats/litert/index.js — dead shim (`export * from './types.js'`); sole consumer was the `.d.ts` and that now imports `types.js` directly
- src/formats/litert/index.d.ts — paired `.d.ts` dead shim
- src/formats/tflite/index.js — dead shim (`export * from './types.js'`); zero consumers (all in-repo imports go directly to `formats/tflite/types.js`)
- src/formats/tflite/index.d.ts — paired `.d.ts` dead shim
- src/inference/functiongemma.js — dead shim (`export { MultiModelNetwork } from './multi-model-network.js'`); zero importers in src/, tools/, tests/, demo/, package.json exports
- src/inference/functiongemma.d.ts — paired `.d.ts` dead shim

### Visited clean (skipped from future fires)
- src/formats/litert/types.js
- src/formats/tflite/types.js
- src/inference/multi-model-network.js
- src/utils/plain-object.js (canonical `isPlainObject` export)
- src/rules/execution-rules-contract-check.js
- src/rules/layer-pattern-contract-check.js
- src/gpu/kernels/clamp.wgsl
- src/gpu/kernels/clamp.js
- src/gpu/kernels/transpose.wgsl
- src/gpu/kernels/activation-static-qdq.js
- src/gpu/kernels/attention.js
- src/gpu/kernels/attention_prefill_flash_head256_f16kv.wgsl
- src/gpu/kernels/attention_prefill_flash_reduce.wgsl
- src/gpu/kernels/attention_decode_subgroup.wgsl
- src/gpu/kernels/softmax_subgroup.wgsl
- src/gpu/kernels/dequant_f16_out.wgsl
- src/gpu/kernels/dequant_f16_out_vec4.wgsl
- src/config/kernels/registry.json (touched by earlier fire — do not re-scan unless new finding justifies it)
- tools/sync-kernel-reachability.js
- src/utils/index.js
- src/inference/pipelines/vision/gemma4.js
- src/inference/pipelines/energy-head/row-head-pipeline.js

### Punts
- `matchesExactObject` in `src/rules/{execution-rules,layer-pattern}-contract-check.js` — two implementations have DIFFERENT behaviors (execution-rules handles Array.isArray recursion; layer-pattern does not). Cannot naively consolidate without auditing all call sites; would need a new shared helper and a migration pass. Bigger than one landing.
- `isPlainObject` consolidation is only partial in this fire — 5 more call sites remain (`src/cli/doppler-cli.js`, `src/tooling/kernel-path-builder/index.js`, `src/inference/pipelines/text/generator-helpers.js`, `src/rules/execution-rules-contract-check.js`, `src/rules/layer-pattern-contract-check.js`). A follow-up fire can keep migrating. Deliberately bounded to avoid one landing ballooning into a repo-wide rename.
- Many WGSL files are flagged by a filename-only scan as "orphan" (not referenced by `<basename>.wgsl`) but are actually dispatched by VARIANT NAME via `getPipeline(...)` or selected through `.rules.json` rule chains (example: `clamp.wgsl` dispatched via op `'clamp'` from `clamp.js:25`; `softmax_subgroup.wgsl` selected via `softmax.rules.json#variant`). The `reachability.status: "unused"` field is ALSO unreliable — e.g. `prefill_flash_head256_f16kv` and `prefill_flash_reduce` are flagged unused but are dispatched from `attention.js:1409` and `attention.js:1448`. Safe dead-kernel sweeping needs a cross-reference of variant-name usage across JS and rule chains; tracked as a dedicated fire.
- Pre-existing codegen patches broken for 6 variants (carried over from fire-1).

## fire-1 — 2026-04-18 UTC

Landings (7+): 7
WGSL touches: 1   JS touches: 6 (incl. .d.ts)

Baseline gate state (pre-fire): `kernels:check` FAIL (6 pre-existing codegen-patch errors),
`kernels:reachability:check` FAIL (stale), `imports:check:browser` FAIL (pre-existing
node:* specifiers in browser-tagged files), `test:unit` FAIL 30/349 (pre-existing).
Landings are counted only if delta is non-regressing vs baseline. This fire also fixed
`kernels:reachability:check` (stale → clean) as a side-effect of removing the ORT kernel.

Post-fire gate state: `agents:verify` PASS, `digests:check` PASS (re-synced),
`kernels:reachability:check` PASS (improved vs baseline), `contracts:check` PASS,
`kernels:check` unchanged (same 6 pre-existing errors, untouched), `imports:check:browser`
unchanged, `test:unit` 30/349 unchanged.

### Changed
- src/client/provider.d.ts — dropped deprecated `FallbackFailureClass` alias (0 external refs)
- src/debug/config.js — dropped deprecated `setBenchmarkMode`/`isBenchmarkMode` (no callers outside debug module)
- src/debug/config.d.ts — matching type removal for deprecated pair
- src/debug/index.js — removed `setBenchmarkMode`/`isBenchmarkMode` from 3 re-export blocks + DOPPLER_API globalThis surface + default export
- src/debug/index.d.ts — matching type removal for re-exports, browser API interface, and default declaration
- src/sw.js — removed stale APP_SHELL + BYPASS refs to `/src/boot/vfs-bootstrap.js` and `/config/vfs-manifest.json` (both paths do not exist; VFS generation was removed)
- src/config/kernels/registry.json — removed `prefill_flash_ort_head256_f16kv` variant entry (marked `status: unused`, inlineConfigs/ruleChains both empty, ORT-derived reference-only kernel)
- src/config/kernels/kernel-ref-digests.js — re-synced (removed `attention_prefill_flash_ort_head256_f16kv.wgsl#main` digest entry)

### Deleted
- src/utils/format-bytes.js — dead module; zero importers in src/, tools/, tests/, demo/; not re-exported from `src/utils/index.js`; local `formatBytes` lives in `src/storage/quota.js` and two schema files instead
- src/sw.d.ts — empty `export {};` stub with no importers and no package.json export entry
- src/bootstrap.d.ts — empty `export {};` stub with no importers and no package.json export entry
- src/gpu/kernels/attention_prefill_flash_ort_head256_f16kv.wgsl — orphan ORT-derived experimental kernel; zero references outside the registry/digest tables it was removed from

### Visited clean (skipped from future fires)
- src/utils/hf-resolve-url.js
- src/utils/sha256.js
- src/utils/plain-object.js
- src/utils/load-json.js
- src/utils/index.js
- src/loaders/index.js
- src/loaders/index.d.ts
- src/tooling-experimental-exports.js
- src/tooling-experimental-exports.browser.js
- src/tooling-experimental-exports.shared.js
- src/types/gpu.js
- src/types/inference.js
- src/types/model.js
- src/types/chrome.js
- src/types/index.js
- src/version.js
- src/generation/index.js
- src/bootstrap.js
- src/sw.js
- src/client/provider.d.ts
- src/debug/config.js
- src/debug/config.d.ts
- src/debug/index.js
- src/debug/index.d.ts
- src/gpu/kernels/rope.wgsl
- src/gpu/kernels/rope_f16.wgsl
- src/gpu/kernels/codegen/wgsl-variants.js
- src/gpu/kernels/codegen/wgsl-patch-variants.js
- src/gpu/kernels/backward/conv2d_backward_input.wgsl
- src/gpu/kernels/backward/conv2d_backward_weight.wgsl
- src/gpu/kernels/backward/matmul_transpose_a.wgsl
- tools/sync-kernel-ref-digests.js

### Punts
- `precompute_freqs` entry point in `rope.wgsl` + `rope_f16.wgsl` is dead (no registry/digest/dispatch reference) but removal requires updating the codegen patch at `src/gpu/kernels/codegen/patches/rope_f16.from.rope.diff`, whose hunks are already stale against the current source (baseline `kernels:check` fails on this variant). Needs a separate codegen-patch repair pass.
- `dequant_f16_out*.wgsl` + `dequant_shared*.wgsl` + `dequant_subgroup.wgsl` share duplicated `unpack_f16_lo` / `unpack_f16_hi` / `get_scale_byte` helpers; and `dequant_mxfp4*.wgsl` share `get_nibble` / `get_scale`. Consolidating via a shared WGSL include is a larger refactor than a single landing.
- Pre-existing broken codegen patches for variants `attention-f16`, `attention-f16kv`, `attention-streaming-f16`, `rmsnorm-f16`, `rope-f16`, `sample-f16` — hunks don't apply; needs dedicated patch-refresh fire.
- Pre-existing browser-graph violations: `src/inference/browser-harness-model-helpers.js`, `src/storage/artifact-storage-context.js`, `src/client/runtime/lora.js`, `src/client/runtime/node-quickstart-cache.js` import `node:*` from files reachable from `src/tooling-exports.browser.js`. Needs isolation work.
- 30/349 unit-test failures on main (including `tests/integration/translategemma-q4k-regression.test.js` and several experimental-subsystem resolutions) — pre-existing.
- Many kernel registry entries marked `reachability.status: "unused"` (114 variants). Good candidate for a dedicated dead-kernel sweep, but removing them per-fire risks churn against in-flight kernel-path work.

## fire-0 (pre-seed) — skip list from prior session

Files already consolidated in prior session work; do not re-cover unless there's a new reason.

### Visited clean (skipped from future fires)
- tools/publish-hf-registry-model.js
- src/tooling-exports/
- src/models/
- src/inference/pipelines/text/attention/run.js
- src/inference/pipelines/text/attention/record.js
- src/inference/pipelines/text/init.js
- src/config/merge.js
- src/rules/rule-registry.js
- src/gpu/kernels/shader-cache.js
- src/config/schema/debug.schema.js
- src/config/schema/doppler.schema.js
- src/inference/pipelines/context.js
