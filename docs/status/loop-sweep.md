## fire-18 — 2026-04-19 UTC

Landings (7+): 7
WGSL touches: 1 (delete + operation+tests removal)   JS touches: 6 (incl. paired .d.ts; 1 delete + 5 demote batches)

Baseline parity vs fire-17: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* (unchanged), `test:unit` 25/349 (unchanged).
No regressions.

### Changed
- src/config/kernels/registry.json — removed entire `attention_decode_optimized` operation (3 variants, all status `unused`, zero production dispatchers; only test-page.js + kernel-suite.js + harness.html referenced it). Operation previously occupied ~1340 lines (1303–2643)
- src/config/kernels/kernel-ref-digests.js — re-synced (249 entries after deletion, down from 252)
- tests/kernels/browser/kernel-suite.js — removed corresponding `attention_decode_optimized` test block (22 lines)
- tests/kernels/browser/test-page.js — removed `runAttentionDecodeOptimized` harness helper (66 lines)
- tests/harness.html — removed duplicate `attention_decode_optimized` test block (16 lines)
- src/cli/doppler-quickstart.js — demoted `runQuickstart` to private (called internally by `main` at line 256; zero external consumers)
- src/cli/doppler-quickstart.d.ts — paired type removal
- src/config/schema/distill-training.schema.js — demoted `DISTILL_STUDENT_GRAPH_MODE_VALUES` to private (used internally for validator; zero external consumers; companion `DISTILL_STAGE_VALUES` stays public)
- src/config/schema/distill-training.schema.d.ts — paired type removal
- src/config/schema/execution-v1.schema.js — demoted `DEFAULT_EXECUTION_V1_PER_LAYER_INPUTS_SESSION` to private (nested inside the larger exported `DEFAULT_EXECUTION_V1_SESSION`; no consumer needs it standalone)
- src/config/schema/execution-v1.schema.d.ts — paired type removal
- src/config/schema/kernel-thresholds.schema.js — demoted `DEFAULT_FUSED_MATMUL_THRESHOLDS` to private (used internally in validator)
- src/config/schema/kernel-thresholds.schema.d.ts — paired type removal
- src/config/schema/ul-training.schema.js — demoted `DEFAULT_UL_LOSS_WEIGHTS` to private (nested in `DEFAULT_UL_TRAINING_CONFIG`)
- src/config/schema/ul-training.schema.d.ts — paired type removal
- src/inference/pipelines/text/generator-runtime.js — demoted `decodeFloatWeights` to private (used 4x internally by `getFinalNormWeights`; zero external)
- src/inference/pipelines/text/generator-runtime.d.ts — paired type removal
- src/inference/pipelines/text/generator-steps.js — demoted `resolveBatchStop` to private (used internally at line 1584; zero external)
- src/inference/pipelines/text/generator-steps.d.ts — paired type removal
- src/inference/pipelines/text/layer-plan.js — demoted `compileLayerPipeline` to private (used by `resolveLayerPipeline` in same file; zero external)
- src/inference/pipelines/text/layer-plan.d.ts — paired type removal
- src/loader/tensors/tensor-loader.js — demoted `loadQ4KMixed` to private (used by the q4k_mixed loader map entry at line 664; zero external)
- src/loader/tensors/tensor-loader.d.ts — paired type removal

### Deleted
- src/gpu/kernels/attention_decode_optimized.wgsl — dead kernel file. 3 variants (`default`, `multihead`, `f16kv`) all marked reachability `unused` with zero rule chains + zero inline configs; no production JS dispatch; only test infrastructure referenced it
- src/inference/pipelines/diffusion/sana-transformer.js — `buildSanaConditioning` export (zero callers; consumers inline the 2 helper calls it composed)
- src/inference/pipelines/diffusion/sana-transformer.d.ts — paired type removal

### Visited clean (skipped from future fires)
- src/gpu/kernels/attention_decode_optimized.wgsl (deleted)
- src/cli/doppler-quickstart.{js,d.ts}
- src/config/schema/distill-training.schema.{js,d.ts}
- src/config/schema/execution-v1.schema.{js,d.ts}
- src/config/schema/kernel-thresholds.schema.{js,d.ts}
- src/config/schema/ul-training.schema.{js,d.ts}
- src/inference/pipelines/diffusion/sana-transformer.{js,d.ts}
- src/inference/pipelines/text/generator-runtime.{js,d.ts}
- src/inference/pipelines/text/generator-steps.{js,d.ts}
- src/inference/pipelines/text/layer-plan.{js,d.ts}
- src/loader/tensors/tensor-loader.{js,d.ts}
- tests/kernels/browser/kernel-suite.js (test block removed; file still live)
- tests/kernels/browser/test-page.js (harness helper removed; file still live)
- tests/harness.html (duplicate test removed; file still live)

### Punts
- The fire-16 duplicate finding (`mergeBindings` byte-identical, `isSlidingLayerType` 3x) still unresolved — consolidation needs a shared utility file, not a surgical single-landing change.
- Many WGSL files are marked `status: unused` with 1 rule-chain ref. These are dispatched through dynamic variant-name selection and LOOK unused to the flat scan but ARE live (same pattern as fire-5's punt on `prefill_flash_head256_f16kv`). Leave as-is.
- Pre-existing codegen patches broken for 6 variants (carried over from fire-1 onward).

## fire-17 — 2026-04-19 UTC

Landings (7+): 7
WGSL touches: 1 (delete)   JS touches: 6 (incl. paired .d.ts; multiple deletes + demotes across 9 files)

Baseline parity vs fire-16: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* (unchanged), `test:unit` 25/349 (unchanged).
No regressions.

### Changed
- src/config/kernels/registry.json — removed `dequant.mxfp4_vec4` variant entry (reachability status `unused`, zero rules, zero JS dispatch of variant name `mxfp4_vec4`; dequant.js line 237 only dispatches `'mxfp4'`)
- src/config/kernels/kernel-ref-digests.js — re-synced (252 entries after deletion)
- src/client/runtime/model-manager.js — demoted `initDoppler` to private (only internal caller at line 270 `loadModel`)
- src/client/runtime/model-manager.d.ts — paired type removal for `initDoppler` + deleted declarations
- src/config/platforms/loader.js — deleted 4 dead functions: `prefersUnifiedMemory`, `getBufferAlignment`, `getResolvedPlatformConfig`, `getMemoryHints` (the last went stale once its 3 external-facing wrappers were removed). Also dropped unused imports (`getRuntimeConfig`) and unused `DEFAULT_PREFER_UNIFIED_MEMORY`
- src/config/platforms/loader.d.ts — paired type removals + unused `MemoryHintsSchema` import
- src/inference/browser-harness-model-helpers.js — demoted `initializeInferenceFromStorage`, `initializeInferenceFromSourcePath`, `resolveHarnessOverride` to private (all used internally by `initializeSuiteModel`; zero external)
- src/inference/browser-harness-model-helpers.d.ts — paired type removals
- src/inference/browser-harness-runtime-helpers.js — demoted `parseReportTimestamp`, `normalizeRuntimeConfigChain` to private
- src/inference/browser-harness-runtime-helpers.d.ts — paired type removals
- src/inference/browser-harness-suite-helpers.js — demoted `calculateRatePerSecond` to private (used locally in buildSuiteSummary)
- src/inference/browser-harness-suite-helpers.d.ts — paired type removal
- src/inference/test-harness.js — demoted `createHttpShardLoader`, `fetchManifest` to private (both used only by `initializeInference` in same file)
- src/inference/test-harness.d.ts — paired type removals

### Deleted
- src/gpu/kernels/dequant_mxfp4_vec4.wgsl — orphan WGSL: sole variant `dequant.mxfp4_vec4` marked unused, no JS dispatch, no rule selects it (dequant.rules.json has `subgroup_vec4` / `shared` / etc., never `mxfp4_vec4`)
- src/client/runtime/model-manager.js — `getDopplerStorageInfo`, `destroyDoppler` (both zero external, zero internal callers)
- src/formats/rdrr/groups.js — 6 dead exports: `getGroup`, `getGroupIds`, `getShardsForGroup`, `getTensorsForGroup`, `getLayerGroupIds`, `getExpertGroupIds` (all zero consumers; the only live exports that remain are `getShardsForExpert`, `getTensorsForExpert`, `getExpertBytes` which handle MoE expert lookups). Dropped orphan imports `sortGroupIds`, `parseGroupExpertIndex` from `./classification.js`
- src/formats/rdrr/groups.d.ts — paired type removals (6 declarations)
- src/formats/rdrr/tensor-config-validator.js — `formatValidationResult` (formatter never called; only `validateTensorConfigConsistency` is used by consumers)
- src/formats/rdrr/tensor-config-validator.d.ts — paired type removal
- src/inference/pipelines/text/kernel-trace.js — `traceStepSync` (async `traceStep` is the only used variant; sync path has zero callers)
- src/inference/pipelines/text/kernel-trace.d.ts — paired type removal
- src/inference/pipelines/text/model-load.js — `resolveAndActivateKernelPath` (zero callers; consumers inline `resolveKernelPathState` + `activateKernelPathState`)
- src/inference/pipelines/text/model-load.d.ts — paired type removal
- src/inference/pipelines/text/embed.js — `validateEmbedding` (debug/trace-only helper, zero callers)
- src/inference/pipelines/text/embed.d.ts — paired type removal
- src/inference/pipelines/text/execution-runtime-builders.js — `resolveFinitenessFallbackKernelPathId` (zero callers; consumers pass the fallback id directly)
- src/inference/pipelines/text/execution-runtime-builders.d.ts — paired type removal

### Visited clean (skipped from future fires)
- src/gpu/kernels/dequant_mxfp4_vec4.wgsl (deleted)
- src/client/runtime/model-manager.{js,d.ts}
- src/config/platforms/loader.{js,d.ts}
- src/formats/rdrr/groups.{js,d.ts}
- src/formats/rdrr/tensor-config-validator.{js,d.ts}
- src/inference/browser-harness-model-helpers.{js,d.ts}
- src/inference/browser-harness-runtime-helpers.{js,d.ts}
- src/inference/browser-harness-suite-helpers.{js,d.ts}
- src/inference/test-harness.{js,d.ts}
- src/inference/pipelines/text/kernel-trace.{js,d.ts}
- src/inference/pipelines/text/model-load.{js,d.ts}
- src/inference/pipelines/text/embed.{js,d.ts}
- src/inference/pipelines/text/execution-runtime-builders.{js,d.ts}

### Punts
- Many demote candidates remain internal (e.g. `runQuickstart`, `decodeFloatWeights`, `compileLayerPipeline`, `loadQ4KMixed`, `DISTILL_STUDENT_GRAPH_MODE_VALUES`, etc.) — all used internally >=2x, zero external. Deferred to keep fire-17 landing count surgical; future fires can keep picking them off one file at a time.
- `mergeBindings` byte-identical duplication across `src/config/kernels/registry.js` + `src/config/schema/kernel-registry.schema.js` — still punted (consolidation reverses layered module boundary).
- `isSlidingLayerType` 3x duplication still unconsolidated — needs a new shared utility file; not a single-landing fit.
- Pre-existing codegen patches broken for 6 variants (carried over from fire-1 onward).

## fire-16 — 2026-04-19 UTC

Landings (7+): 7
WGSL touches: 1 (delete)   JS touches: 6 (incl. paired .d.ts; 1 dead-function delete + 5 demotion batches)

Baseline parity vs fire-15: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* (unchanged), `test:unit` 25/349 (unchanged).
No regressions.

### Changed
- src/config/kernels/registry.json — removed `moe_gather.gather_vec4` variant entry (reachability status `unused`, entry-point `gather_tokens_vec4` only selectable via suffix `_vec4` which no rule generates; moe.js rule yields `_f16` or `""` only)
- src/config/kernels/kernel-ref-digests.js — re-synced (253 entries after deletion)
- src/gpu/kernels/turboquant-codebook.js — demoted 11 exports to private: `computeMaxLloydCodebook`, `getCodebook`, `getRotationMatrix`, `getQJLMatrix`, `generateRotationMatrix`, `generateQJLMatrix`, `computeOutlierFraction`, `uploadRotationMatrix`, `uploadCodebook`, `ROTATION_SEED`, `QJL_SEED`. All internal-only; fire-10 explicitly punted this batch under "deferred to keep landings surgical". Live exports `retainTurboQuantSharedBuffers` + `computePackedStride` stay public
- src/inference/pipelines/text/layer.js — demoted `processLayerCPU` to private (only called internally at line 452 from `processLayer` fallback; zero external consumers)
- src/inference/pipelines/text/layer.d.ts — paired type removal
- src/debug/stats.js — demoted 3 internal-only exports: `removeOutliersIQR`, `sampleStdDev`, `confidenceInterval95`. Public API stays: `percentile`, `median`, `computeSampleStats`, `computeArrayStats`, `computeBasicStats`. Fire-14 held these on a stale reference to `experimental/distribution/p2p-observability.js`; re-scan confirms that file now only uses `percentile`
- src/debug/stats.d.ts — paired type removals for the 3 demoted helpers
- src/inference/pipelines/text/ops.js — demoted `isDecodeBuffer` to private (used internally by `releaseOrTrack` at line 34; zero external)
- src/inference/pipelines/text/ops.d.ts — paired type removal
- src/config/kernels/registry.js — demoted 5 internal-only exports: `getRegistrySync`, `getOperation`, `getVariant`, `getVariantNames`, `isVariantAvailable`. External consumers (`setRegistryUrl`, `getRegistry`, `clearRegistryCache`, `mergeBindings`, `resolveKernelConfig`) stay public
- src/config/kernels/registry.d.ts — paired type removals
- src/inference/pipelines/text/config.js — demoted 3 internal-only exports: `validateChatTemplateType`, `hasManifestInference`, `toParsedConfigFromMerged`. All used locally only; `validateRequiredInferenceFields` stays public (external consumers)
- src/inference/pipelines/text/config.d.ts — paired type removals

### Deleted
- src/gpu/kernels/moe_gather_vec4.wgsl — orphan WGSL: registry status `unused`, entry point `gather_tokens_vec4` not dispatched; moe.js only uses `gather` / `gather_f16` suffix forms. Same clean-delete pattern as fire-15 (fused_matmul_residual_f16.wgsl)
- src/config/kernels/registry.js — `getAvailableVariants` export removed (dead function; zero external consumers AND zero internal callers; was a thin wrapper over `getVariantNames` + `isVariantAvailable`)
- src/config/kernels/registry.d.ts — paired type removal for `getAvailableVariants`

### Visited clean (skipped from future fires)
- src/gpu/kernels/moe_gather_vec4.wgsl (deleted)
- src/gpu/kernels/turboquant-codebook.js (re-visited — fire-10 punt explicitly scheduled this demote batch for a future fire)
- src/inference/pipelines/text/layer.{js,d.ts}
- src/debug/stats.{js,d.ts} (re-visited — fire-14 stale reference to experimental consumer cleared)
- src/inference/pipelines/text/ops.{js,d.ts}
- src/config/kernels/registry.{js,d.ts}
- src/inference/pipelines/text/config.{js,d.ts}

### Punts
- `mergeBindings` is byte-identical in `src/config/kernels/registry.js` and `src/config/schema/kernel-registry.schema.js`. Consolidating (e.g., importing one from the other) adds a circular-looking module boundary between runtime loader and schema helpers. Defer.
- `isSlidingLayerType` is duplicated 3x: exported in `layer.js` + `dispatch-params.js`, private in `init.js` (SKIP-listed) + `kv-cache/mixed-geometry.js`. Consolidating requires a new shared utility file (e.g., `attention/layer-type-utils.js`); not a single-landing fit because dispatch-params.js sits under attention/ and can't simply import from `../layer.js` without reversing the dependency direction. Defer.
- Dead `.d.ts` types (e.g., `AdamOptions`, `BiasAddBackwardOptions`, `KernelRuntimeOptions`, `CheckHotVocabStopParams`) are only used for parameter typing in the same file — externally dead but internally documentary. Inlining into function signatures is a readability regression, so keep as-is.
- Pre-existing codegen patches broken for 6 variants (carried over from fire-1 onward).

## fire-15 — 2026-04-19 UTC

Landings (7+): 7
WGSL touches: 1 (delete)   JS touches: 6 (incl. paired .d.ts; 2 deletes + 2 demotion files + 2 dedup consolidations)

Baseline parity vs fire-14: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* (unchanged), `test:unit` 25/349
(improved from 26/349 — one pre-existing failure resolved as side-effect of
kernel-validator delete; no new failures).

### Changed
- src/inference/pipelines/text/execution-plan.js — demoted `PRIMARY_EXECUTION_PLAN_ID` + `FINITENESS_FALLBACK_EXECUTION_PLAN_ID` to private (heavy internal use at lines 153, 156, 231, 277, 321, 332, 350, 358; zero external consumers; cleared the fire-14 punt)
- src/inference/pipelines/text/execution-plan.d.ts — paired type removals
- src/inference/pipelines/text/attention/attn-config.js — demoted `ATTN_CONFIG_REQUIRED_FIELDS` to private (only used internally at line 40; zero external; cleared fire-14 punt)
- src/inference/pipelines/text/attention/attn-config.d.ts — paired type removal
- src/inference/pipelines/text/weights.js — removed 2 dead exports (`BatchBufferTracker` class, `createWeightBufferHelpers` function; both zero references anywhere) + demoted `isLayerWeights` and `getGPUWeightBuffer` to private (used internally only)
- src/inference/pipelines/text/weights.d.ts — paired type removals
- src/inference/pipelines/text/logits/cpu.js — replaced duplicate `f16ToF32` implementation with import + re-export from `src/loader/dtype-utils.js` (canonical source; byte-identical implementation). Preserves `logits/index.js` barrel re-export path for downstream consumers
- src/inference/pipelines/text/debug-utils/utils.js — same dedup: replaced duplicate `f16ToF32` with import + re-export from `src/loader/dtype-utils.js`. Preserves `debug-utils/index.js` barrel re-export
- src/config/kernels/registry.json — removed `fused_ffn.matmul_residual_tiled_f16` variant entry (reachability status `unused`, zero rule chains, zero JS dispatch of the variant name — sole consumer of the WGSL file being deleted)
- src/config/kernels/kernel-ref-digests.js — re-synced (254 entries after deletion)

### Deleted
- src/rules/kernels/kernel-validator.js — dead module; `KernelValidator` class had zero external importers and nobody imports the module itself. (24 lines .d.ts + 160 lines .js, all dead)
- src/rules/kernels/kernel-validator.d.ts — paired .d.ts delete
- src/gpu/kernels/fused_matmul_residual_f16.wgsl — 105-line WGSL file with sole variant (`matmul_residual_tiled_f16`) marked unused + zero JS dispatchers. Cross-ref: not a codegen target (no `.from.` patch produces it), only referenced in registry.json + kernel-ref-digests.js. Same clean-delete pattern as fire-11 (dequant_q8_0.wgsl) and fire-1 (attention_prefill_flash_ort_head256_f16kv.wgsl)

### Visited clean (skipped from future fires)
- src/inference/pipelines/text/execution-plan.{js,d.ts}
- src/inference/pipelines/text/attention/attn-config.{js,d.ts}
- src/inference/pipelines/text/weights.{js,d.ts} (re-visited from fire-14 inventory; new dead-export findings `BatchBufferTracker` + `createWeightBufferHelpers`)
- src/rules/kernels/kernel-validator.{js,d.ts} (deleted)
- src/inference/pipelines/text/logits/cpu.js
- src/inference/pipelines/text/debug-utils/utils.js
- src/gpu/kernels/fused_matmul_residual_f16.wgsl (deleted)

### Punts
- `isPlainObject` duplicate in `src/tooling/hf-registry-utils.js` (still defines its own copy; canonical lives in `src/utils/plain-object.js`). hf-registry-utils.js is on skip list from fire-6; swap has 6 internal call sites + .d.ts type. Defer as part of the broader `isPlainObject` consolidation fire-5 already tracked.
- 5 `export const format*Chat` aliases were re-visited in fire-14; no new alias patterns found in this fire's non-skiplist JS sweep. The scan surface is mostly clean.
- `formatBytes` has 3 DIFFERENT implementations across `emulation.schema.js`, `units.schema.js`, `storage/quota.js` (decimal vs binary units, different formats). Cannot naively consolidate — each has distinct semantics and consumers; needs a design decision on unit convention rather than a surgical DRY fix.
- `f16ToF32` consolidation (this fire) only targeted the 2 byte-identical duplicates; the canonical `loader/dtype-utils.js` implementation stays exported.
- Pre-existing codegen patches broken for 6 variants (carried over from fire-1 onward).

## fire-14 — 2026-04-19 UTC

Landings (7+): 7
WGSL touches: 1   JS touches: 6 (incl. paired .d.ts; 3 deletes + 3 demotion batches)

Baseline parity vs fire-13: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* (unchanged), `test:unit` 26/349 (unchanged).
No regressions.

### Changed
- src/gpu/kernels/backward/layernorm_backward.wgsl — removed dead `override PARAMS_IS_F16` (the inline comment on line 30 explicitly marked it "unused for now as we read f32"; zero references in kernel body)
- src/config/kernels/kernel-ref-digests.js — re-synced after WGSL edit
- src/inference/pipelines/vision/index.js — removed dead export `mergeVisualTokens` (~85 lines; zero external consumers; no paired .d.ts). DeepStack injection happens in decoder layer loop, not via this helper
- src/debug/trace.js — removed dead export `clearTraceEntries` (zero external consumers; stale doc comment updated to drop reference)
- src/gpu/kernels/attention.js — demoted `executeFlashAttentionPrefill` + `executeOrtFlashAttentionPrefill` to private (both only called internally at lines 1097 and 1056; zero external consumers). The fire-9 punt explicitly flagged these as internal-only but deferred due to skip-list discipline; the skip entry is from fire-1 (13 fires ago) and fire-9 re-established interest
- src/config/param-categories.js — removed dead export `getParamCategory` (helper `PARAM_CATEGORIES[name] ?? null`; zero consumers; the dict + `CategoryRules` still exported and used by `param-validator.js`)
- src/config/param-categories.d.ts — paired type removal
- src/inference/pipelines/text/chat-format.js — removed 5 dead `export const format*Chat = format*` aliases (`formatGemmaChat`, `formatGemma4Chat`, `formatLlama3Chat`, `formatGptOssChat`, `formatTranslateGemmaChat`); zero external consumers. Fire-12 cleaned function exports here; this fire caught the arrow/alias exports that the earlier `export function`-only scan missed (valid re-visit per skip-list rule)
- src/inference/pipelines/text/chat-format.d.ts — paired type removals (5 declare functions)
- src/inference/network-evolution.js — demoted `mutateGenome` + `crossoverGenome` to private (both are arrow-`const` exports used only internally at line 67; zero external consumers; `evolveNetwork` remains exported and is the tested surface)
- src/inference/network-evolution.d.ts — paired type removals for the 2 helpers

### Visited clean (skipped from future fires)
- src/gpu/kernels/backward/layernorm_backward.wgsl
- src/inference/pipelines/vision/index.js
- src/debug/trace.js
- src/gpu/kernels/attention.js (re-visited — fire-9 re-established interest on these specific demotions)
- src/config/param-categories.{js,d.ts}
- src/inference/pipelines/text/chat-format.js (re-visited — new finding: arrow-alias exports not caught by prior function-only scan)
- src/inference/network-evolution.{js,d.ts}

### Punts
- `getLogHistory`, `printLogSummary`, `getDebugSnapshot` in `src/debug/history.js` — appear externally unreferenced but ARE exposed on the browser `DOPPLER_API` global + debug default export, so removing them silently breaks the browser console surface. Hold.
- `createEscalationPolicy` + `validateCaptureConfig` in `src/debug/capture-policy.js` — consumed by `src/inference/pipelines/text/generator.js` and `tests/inference/operator-diffing.test.js`. Not dead; keep exported.
- `percentile`, `removeOutliersIQR`, `sampleStdDev`, `confidenceInterval95` in `src/debug/stats.js` — consumed by `src/experimental/distribution/p2p-observability.js` (experimental consumer, off-limits per skip discipline). Hold.
- `ROTATION_SEED` / `QJL_SEED` in `src/gpu/kernels/turboquant-codebook.js` and `PRIMARY_EXECUTION_PLAN_ID` / `FINITENESS_FALLBACK_EXECUTION_PLAN_ID` in `src/inference/pipelines/text/execution-plan.js` — used only internally; demote candidates for a future fire (scope-bounded to keep diff < 100 lines per landing).
- `ATTN_CONFIG_REQUIRED_FIELDS` in `src/inference/pipelines/text/attention/attn-config.js` — used only internally; demote candidate for a future fire.
- `parseConfigJsonText` / `parseTokenizerConfigJsonText` / `parseTokenizerJsonText` in `src/formats/tokenizer/types.js` — still held (experimental consumer, same as fire-11 punt).
- Pre-existing codegen patches broken for 6 variants (carried over from fire-1 onward).

## fire-13 — 2026-04-19 UTC

Landings (7+): 7
WGSL touches: 2   JS touches: 5 (incl. 2 dead-function removals + 3 demotion batches)

Baseline parity vs fire-12: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* (unchanged), `test:unit` 26/349 (unchanged).
No regressions.

### Changed
- src/gpu/kernels/kv_quantize_turboquant.wgsl — removed unused `const MAX_WORKGROUP_SIZE` (re-visited; fire-2 cleaned `NUM_CENTROIDS` but missed this adjacent dead const)
- src/gpu/kernels/kv_quantize_turboquant_prod.wgsl — same pattern
- src/config/kernels/kernel-ref-digests.js — re-synced after WGSL edits
- src/config/schema/emulation.schema.js — demoted 16 exports to private: `H100_GPU_SPEC`, `H200_GPU_SPEC`, `B200_GPU_SPEC`, `GRACE_CPU_SPEC`, `NVLINK_4_SPEC`, `NVLINK_5_SPEC`, `GH200_TOPOLOGY`, `GH200_NVL2_TOPOLOGY`, `GB200_8GPU_TOPOLOGY`, `GB200_NVL72_TOPOLOGY`, `TP2_PARALLELISM_CONFIG`, `TP8_PARALLELISM_CONFIG`, `DEFAULT_TIMING_SCALING`, `calculateTotalVram`, `calculateTotalCpuMemory`, `formatBandwidth`. All used internally; zero external consumers. The `DEFAULT_*` aliases (GH200_GPU_SPEC, GH200_CPU_SPEC, NVLINK_SPEC, NVLINK_C2C_SPEC, PARALLELISM_CONFIG, EMULATION_CONFIG) stay exported as public API aliases + the live `formatBytes`, `getChipProfile`, `createEmulationConfig`
- src/config/schema/kvcache.schema.js — removed dead function `validateKvCacheDtype` (declared once, zero callers anywhere)
- src/config/schema/moe.schema.js — removed dead function `validateMoeRoutingConfig` (same)
- src/cli/cli-output.js — demoted 6 dead exports to private: `quoteOneLine`, `quoteOneLineOrStructured`, `normalizeBenchMetrics`, `printGpuPhases`, `printMemoryReport`, `printExecutionContractSummary` (all used internally, zero external)
- src/cli/doppler-serve.js — demoted `startServer` to private (used internally at line 379, zero external)

### Visited clean (skipped from future fires)
- src/gpu/kernels/kv_quantize_turboquant.wgsl (re-visited)
- src/gpu/kernels/kv_quantize_turboquant_prod.wgsl (re-visited)
- src/config/schema/emulation.schema.js
- src/config/schema/kvcache.schema.js
- src/config/schema/moe.schema.js
- src/cli/cli-output.js (re-visited — fire-0 pre-seed marked `cli-output.js` conceptually but never removed anything)
- src/cli/doppler-serve.js

### Punts
- emulation.schema.js `DEFAULT_*` aliases still exported but only consumed within the schema barrel (`DEFAULT_GH200_GPU_SPEC` → re-exported from `src/config/schema/index.js`). Consumers outside schema/ weren't surveyed; keep exports for safety.
- Pre-existing codegen patches broken for 6 variants (carried over).
- Scan false-positive rate getting higher as fires compound — many "dead exports" at shallow inspection turn out to be used internally only; demotion is safer than removal.

## fire-12 — 2026-04-19 UTC

Landings (7+): 7
WGSL touches: 1   JS touches: 6 (incl. 3 full-module deletes, 2 shim-pair deletes, 2 demotions)

Baseline parity vs fire-11: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* (unchanged), `test:unit` 26/349 (unchanged).
No regressions.

### Changed
- src/gpu/kernels/dequant_shared_vec4.wgsl — removed unused `const NUM_SUBBLOCKS` (dead doc-only constant, never referenced; same pattern as fire-3's cleanup of the non-vec4 variant)
- src/config/kernels/kernel-ref-digests.js — re-synced after WGSL edit
- src/inference/pipelines/text/chat-format.js — removed dead export `formatQwenChat` (zero external consumers; `formatQwen` is still internally defined for future use)
- src/inference/pipelines/text/generator-decode-policy.js — demoted `assertResolvedKVDtype` to private (used internally at lines 212/219, zero external consumers)
- src/inference/pipelines/text/probes.js — demoted `getCanonicalStageName` to private (used internally at line 124, zero external)

### Deleted
- src/formats/gguf/index.{js,d.ts} — pure re-export shim (`export * from './types.js'`); zero consumers (all imports go to `./types.js` directly)
- src/formats/safetensors/index.{js,d.ts} — same pattern, zero consumers
- src/config/kernels/kernel-ref.{js,d.ts} — dead module; 5 exports (`KERNEL_REF_VERSION`, `getKernelRefContentDigest`, `buildKernelRefFromKernelEntry`, `buildLegacyKernelRefFromKernelEntry`, `isKernelRefBoundToKernel`) all had zero importers. (Note: `kernel-ref-digests.js` is the hyphenated sibling and remains live.)
- src/inference/pipelines/text/resolve-session-flag.js — dead module; both exports (`resolveSessionFlag`, `resolveLargeWeightOverrides`) had zero consumers. No paired .d.ts.

### Visited clean (skipped from future fires)
- src/formats/gguf/types.{js,d.ts} (live via direct imports)
- src/formats/safetensors/types.{js,d.ts} (live via direct imports)
- src/config/kernels/kernel-ref.{js,d.ts} (deleted)
- src/inference/pipelines/text/resolve-session-flag.js (deleted)
- src/inference/pipelines/text/chat-format.js
- src/inference/pipelines/text/generator-decode-policy.js
- src/inference/pipelines/text/probes.js
- src/gpu/kernels/dequant_shared_vec4.wgsl

### Punts
- `src/formats/rdrr/index.js` is a multi-line re-export barrel (`export * from './types.js'`, etc.) with live consumers; collapsing requires migrating each consumer — not a single-landing fit.
- Pre-existing codegen patches broken for 6 variants (carried over).
- Ongoing file-count-based scan produced a long false-positive list in earlier fires; fire-12's scan used a stricter consumer-file exclusion pattern (excluding the defining file + paired .d.ts) which worked for the landings here. Still some false positives — watch for test-only consumers (fire-10 lesson) and experimental/ consumers (fire-11 lesson).

## fire-11 — 2026-04-19 UTC

Landings (7+): 7
WGSL touches: 2   JS touches: 5 (incl. paired .d.ts + 1 WGSL delete + 1 full-file-chain cleanup)

Baseline parity vs fire-10: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* (unchanged), `test:unit` 26/349 (unchanged).
No regressions, after reverting a test-breaking `parseConfigJsonText` demotion mid-fire
(experimental/browser/safetensors-parser-browser.js consumes it — experimental/ imports
are legitimate external consumers but were not surfaced by my src/-only scan).

### Changed
- src/gpu/kernels/matmul_gemv_subgroup.wgsl — removed unused `const MULTICOL_MAX_SUBGROUPS` (declared once at line 155, never referenced in shader body)
- src/config/kernels/registry.json — removed `q8_0_f16out` variant (WGSL file + sole dispatcher both deleted in this fire)
- src/config/kernels/kernel-ref-digests.js — re-synced (255 entries after removal)
- src/gpu/kernels/moe.js — removed 4 dead exports (`recordTopK`, `recordMoEGather`, `recordScatterAdd`, `recordScatterAddDynamic`); zero external/internal callers
- src/gpu/kernels/moe.d.ts — paired type removals
- src/formats/tflite/types.js — demoted 5 `TFLITE_TENSOR_*` consts to private (used internally at lines 102, 585–619, zero external consumers); kept `TFLITE_FILE_IDENTIFIER` exported (live via litert-package-runtime.js)
- src/formats/tflite/types.d.ts — paired type removals
- src/gpu/tensor.js — removed 3 dead exports (`assertDtype`, `assertShape`, `dtypesMatch`); zero external consumers
- src/gpu/tensor.d.ts — paired type removals
- src/gpu/kernels/dequant.js — removed `dequantizeQ8_0` function (fully dead; sole dispatcher for Q8_0 WGSL) + dropped orphan imports `Q8_0_BLOCK_BYTES`, `Q8_0_BLOCK_SIZE`
- src/gpu/kernels/dequant.d.ts — paired type removal
- src/gpu/kernels/index.js, index.d.ts — removed `dequantizeQ8_0` re-exports
- src/tooling/source-artifact-adapter.js — demoted 4 helper functions to private (`normalizeSourceArtifactKind`, `assertSupportedSourceDtypes`, `resolveSourceRuntimeComputePrecision`, `resolveSourceRuntimeModelIdHint`); kept the 5 `SOURCE_ARTIFACT_KIND_*` consts exported because `.d.ts` type aliases (`SourceArtifactKind`, `DirectSourceRuntimeKind`) reference them via `typeof`
- src/tooling/source-artifact-adapter.d.ts — paired type removals for the 4 helpers
- src/gpu/device.js — removed dead export `isPlatformInitialized`; zero external consumers
- src/gpu/device.d.ts — paired type removal

### Deleted
- src/gpu/kernels/dequant_q8_0.wgsl — orphan WGSL after its only JS dispatcher (`dequantizeQ8_0`) was removed

### Visited clean (skipped from future fires)
- src/gpu/kernels/moe.{js,d.ts}
- src/formats/tflite/types.{js,d.ts}
- src/gpu/tensor.{js,d.ts}
- src/gpu/kernels/dequant.{js,d.ts}
- src/gpu/kernels/dequant_q8_0.wgsl (deleted)
- src/gpu/kernels/matmul_gemv_subgroup.wgsl
- src/tooling/source-artifact-adapter.{js,d.ts}
- src/gpu/device.{js,d.ts} (re-visited — new dead-export finding; fire-3 touched errors/index redirect only)
- src/experimental/browser/safetensors-parser-browser.js (verified as external consumer of `parseConfigJsonText` — do NOT demote)
- src/formats/tokenizer/types.{js,d.ts} (demotion reverted)

### Punts
- `parseConfigJsonText` and `parseTokenizerConfigJsonText` in `src/formats/tokenizer/types.js` remain exported — only consumer is `src/experimental/browser/safetensors-parser-browser.js`. Since experimental code is off-limits for modifications unless straight-delete, we can't rewire the consumer. Hold as-is.
- `finalizeBrowserRelayResponse` in `src/tooling/node-browser-command-runner.js` — contract-test consumer (fire-10 lesson).
- Many additional `DEAD_EXPORT` candidates in gpu/kernels (runFusedGateUpGelu, recordFusedGateUpGelu — actually LIVE via dense.js dynamic import), backward/utils.js (runMatmulBackwardDx — LIVE via matmul_backward.js). False positives from file-count-based scan; scan should be refined to count consumer files only.
- Pre-existing codegen patches broken for 6 variants (carried over from fire-1 onward).

## fire-10 — 2026-04-19 UTC

Landings (7+): 7
WGSL touches: 1   JS touches: 6 (incl. paired .d.ts + 1 full-file delete)

Baseline parity vs fire-9: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* (unchanged), `test:unit` 26/349 (unchanged).
No regressions.

Near-miss this fire: attempted `finalizeBrowserRelayResponse` demotion broke
`tests/integration/node-browser-command-relay-contract.test.js` (my export-dead scan counts
files but the test imports the symbol); reverted immediately and replaced with
`extractTensorEntriesFromManifest` demotion instead. Lesson: always grep test refs before
removing/demoting.

### Changed
- src/gpu/kernels/matmul_gemv_subgroup.wgsl — removed unused `const MULTICOL_MAX_SUBGROUPS` (declared once at line 155, never referenced in shader body)
- src/config/kernels/kernel-ref-digests.js — re-synced after WGSL edit
- src/gpu/kernels/turboquant-codebook.js — removed 2 truly-dead exports (`resolveOutlierConfig`, `computePackFactor`) with zero refs anywhere
- src/gpu/kernels/logit-merge.js — removed dead export `mergeLogits` (zero external consumers) + demoted `getLogitMergeKernel` to private (only used internally by `mergeMultipleLogits`)
- src/gpu/kernels/logit-merge.d.ts — paired type removals
- src/tooling/command-api-helpers.js — demoted `asOptionalCacheMode` + `asOptionalLoadMode` to private (used internally at lines 239/240, zero external)
- src/tooling/command-api-helpers.d.ts — paired type removals
- src/tooling/command-envelope.js — demoted `TOOLING_ERROR_CODE_FALLBACK` to private (used internally at lines 71/90, zero external)
- src/tooling/command-envelope.d.ts — paired type removal
- src/tooling/conversion-config-materializer.js — demoted `extractTensorEntriesFromManifest` to private (used internally, zero external)
- src/tooling/conversion-config-materializer.d.ts — paired type removal

### Deleted
- src/gpu/kernels/fused-matmul-residual.js — dead JS dispatcher; zero importers; the `fusedMatmulResidual` rule key in rule-registry refers to a variant string, not this file

### Visited clean (skipped from future fires)
- src/gpu/kernels/matmul_gemv_subgroup.wgsl
- src/gpu/kernels/turboquant-codebook.js (2 truly-dead + 10 internal-only remain — deferred; see punts)
- src/gpu/kernels/logit-merge.{js,d.ts}
- src/tooling/command-api-helpers.{js,d.ts}
- src/tooling/command-envelope.{js,d.ts}
- src/tooling/conversion-config-materializer.{js,d.ts}
- src/tooling/node-browser-command-runner.js (finalizeBrowserRelayResponse: export KEPT — test-only consumer)

### Punts
- `src/gpu/kernels/turboquant-codebook.js` has ~10 more exports that are used internally but have zero external consumers (getCodebook, getRotationMatrix, getQJLMatrix, generateRotationMatrix, generateQJLMatrix, computeOutlierFraction, uploadRotationMatrix, uploadCodebook, ROTATION_SEED, QJL_SEED, computeMaxLloydCodebook). Could all be demoted to private in one larger fire. Deferred to keep landings surgical.
- `src/tooling/source-artifact-adapter.js` has 9 similarly-dead internal-only exports (SOURCE_ARTIFACT_KIND_SAFETENSORS/GGUF/TFLITE/LITERT_TASK/LITERTLM, normalizeSourceArtifactKind, assertSupportedSourceDtypes, resolveSourceRuntimeComputePrecision, resolveSourceRuntimeModelIdHint). Same pattern — deferred as batch.
- `src/tooling/kernel-path-builder/index.js` flagged 4 dead builder exports but file is long-standing git-dirty; deferred.
- `src/gpu/kernels/backward/utils.js` exports `runMatmulBackwardDx`/`recordMatmulBackwardDx` flagged dead. Backward subsystem is experimental-adjacent; need careful audit before removing. Deferred.
- Pre-existing codegen patches broken for 6 variants (carried over from fire-1).

## fire-9 — 2026-04-19 UTC

Landings (7+): 7
WGSL touches: 1   JS touches: 6 (incl. 4 full-file deletes, paired .d.ts edits)

Baseline parity vs fire-8: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* (unchanged), `test:unit` 26/349 (unchanged).
Reachability count dropped 263 → 262 as a side-effect of the orphan WGSL removal.

### Changed
- src/config/kernels/registry.json — removed `rmsnorm_matmul_tiled_f16` variant (wgsl file + sole dispatcher both deleted in this fire)
- src/config/kernels/kernel-ref-digests.js — re-synced (256 entries after removal)
- src/gpu/kernels/dispatch.js — removed 6 dead exports (`dispatchMultiBindGroup`, `calculateWorkgroups1D/2D/3D`, `dispatchAdvanced`, `dispatchBatch`), zero external consumers
- src/gpu/kernels/dispatch.d.ts — paired type removals
- src/gpu/kernels/constants.js — removed 2 dead exports (`PERFORMANCE`, `alignSize`), zero external consumers
- src/gpu/kernels/constants.d.ts — paired type removals + cleanup of stale declarations (`DTYPE_SIZES`, `DType`, `getDtypeSize`, `calculateBufferSize`) that referenced symbols already moved/deleted from the `.js` file
- src/tooling/lean-execution-contract-runner.js — demoted `resolveLeanBinary` + `runLeanCheck` from `export` to private; used internally, zero external consumers
- src/tooling/lean-execution-contract-runner.d.ts — paired type removals

### Deleted
- src/gpu/kernels/fused-rmsnorm-q4-widetile.js — dead JS dispatcher with zero importers (the `q4_fused_rmsnorm_widetile` variant is dispatched via `matmul.js`, not this file)
- src/gpu/kernels/fused-matmul-q4-widetile-residual.js — dead JS dispatcher with zero importers (the `q4_fused_widetile_residual` variant is dispatched via `matmul.js`)
- src/gpu/kernels/fused-rmsnorm-matmul.js — dead JS dispatcher with zero importers (only caller of the `rmsnorm_matmul_tiled_f16` variant; removing it orphaned the WGSL)
- src/gpu/kernels/fused_rmsnorm_matmul_tiled_f16.wgsl — orphan WGSL (its only JS dispatcher was the file above)

### Visited clean (skipped from future fires)
- src/gpu/kernels/dispatch.{js,d.ts}
- src/gpu/kernels/constants.{js,d.ts}
- src/gpu/kernels/matmul.js (verified dispatches `q4_fused_rmsnorm_widetile` + `q4_fused_widetile_residual` directly)
- src/gpu/kernels/matmul-selection.js
- src/gpu/kernels/fused_matmul_q4_widetile_residual.wgsl (verified live via matmul.js)
- src/gpu/kernels/fused_rmsnorm_q4_widetile.wgsl (verified live via matmul.js)
- src/tooling/lean-execution-contract-runner.{js,d.ts}
- src/tooling/lean-execution-contract.js (verified live via 2 tools/ callers)
- tools/lean-execution-contract-config-sweep.js, tools/lean-execution-contract-sweep.js

### Punts
- Additional dead-export candidates still flagged but deferred: `src/tooling/command-api-helpers.js` (`asOptionalCacheMode`, `asOptionalLoadMode`), `src/tooling/source-artifact-adapter.js` (8+ unused `SOURCE_ARTIFACT_KIND_*` + helpers), `src/tooling/node-browser-command-runner.js` (`finalizeBrowserRelayResponse`), `src/tooling/conversion-config-materializer.js` (`extractTensorEntriesFromManifest`), `src/tooling/command-envelope.js` (`TOOLING_ERROR_CODE_FALLBACK`), `src/tooling/kernel-path-builder/index.js` (4 builder exports). Need per-module audit before bulk removal.
- `executeFlashAttentionPrefill` + `executeOrtFlashAttentionPrefill` in `src/gpu/kernels/attention.js` are effectively internal-only but `attention.js` is in the long-standing skip list; deferred.
- Pre-existing codegen patches broken for 6 variants (carried over from fire-1 onward).

## fire-8 — 2026-04-19 UTC

Landings (7+): 7
WGSL touches: 2   JS touches: 5 (incl. paired .d.ts)

Baseline parity vs fire-7: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* (unchanged), `test:unit` 26/349 (unchanged).
No regressions.

### Changed
- src/gpu/kernels/fused_ffn_q4k.wgsl — removed dead helper `fn get_q4(...)` (declared once, never called in file)
- src/gpu/kernels/fused_ffn_q4k_f16.wgsl — same
- src/config/kernels/kernel-ref-digests.js — re-synced after WGSL edits
- src/memory/buffer-pool.js — removed 4 dead public exports (`safeRelease`, `createBufferPool`, `createUploadBuffer`, `withBuffer`) + demoted `unmarkPersistentBuffer` from `export` to private (still used internally by `PersistentBufferSet` class)
- src/memory/buffer-pool.d.ts — paired type removals
- src/converter/quantization-info.js — demoted 3 dead exports to private (`validateQuantType`, `normalizePerLayerEmbeddingQuant`, `buildVariantTag`) — all used internally by `buildQuantizationInfo`/`resolveManifestQuantization`, zero external refs
- src/converter/quantization-info.d.ts — paired type removals
- src/inference/runtime-model.js — removed 2 dead exports (`createRuntimeModelFromManifest`, `isRuntimeModelContract`) with zero external refs
- src/inference/runtime-model.d.ts — paired removals
- src/inference/browser-harness.js — removed 4 dead exports (`clearTrainingSuiteModule`, `saveBrowserReport`, `runBrowserHarness`, and the orphaned `initializeBrowserHarness` whose only caller was the deleted `runBrowserHarness`). Verified zero external callers via grep against src/, tools/, tests/, demo/ and confirmed public surface (`src/tooling-exports.shared.d.ts`) only re-exports `applyRuntimeProfile` + `runBrowserSuite`.
- src/inference/browser-harness.d.ts — paired removals
- src/gpu/kernel-tuner/cache.js — removed dead export `clearOnDeviceReset` (zero callers)

### Visited clean (skipped from future fires)
- src/memory/buffer-pool.js, buffer-pool.d.ts
- src/converter/quantization-info.js, quantization-info.d.ts
- src/converter/quantizer.js (verified — `quantizeToInt4PerRowSymmetric`/`dequantizeInt4PerRowSymmetric` live via core + tests)
- src/inference/runtime-model.js, runtime-model.d.ts
- src/inference/browser-harness.js, browser-harness.d.ts
- src/gpu/kernel-tuner/cache.js
- src/gpu/kernels/fused_ffn_q4k.wgsl, fused_ffn_q4k_f16.wgsl
- src/loader/embedding-loader.js, shard-resolver.js, shard-cache.js, loader-state.js, manifest-config.js (all exports live under 4+ refs)
- src/memory/capability.js, heap-manager.js, unified-detect.js, address-table.js (no dead exports found)

### Punts
- Many additional `DEAD_EXPORT` candidates flagged by scan across `src/inference/browser-harness-model-helpers.js`, `src/inference/browser-harness-suite-helpers.js`, `src/inference/pipelines/diffusion/sana-transformer.js`, `src/rules/kernels/kernel-validator.js`. Deferred to future fires to keep landings ≤100 LOC diff each.
- `MAX_WORKGROUP_SIZE`, `MAX_KV_LEN`, `MAX_HEAD_DIM` constants duplicated across 14-47 WGSL files. WGSL lacks a first-class include mechanism in this codebase — consolidating would require a meaningful codegen/preprocessing layer. Scoped too large for a per-fire landing.
- Pre-existing codegen patches broken for 6 variants (carried over from fire-1 onward).

## fire-7 — 2026-04-19 UTC

Landings (7+): 7
WGSL touches: 1   JS touches: 6 (incl. 4 full-file deletes, 1 paired .d.ts, 5 migrations)

Baseline parity vs fire-6: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* (unchanged), `test:unit` 26/349 (unchanged).
No regressions.

### Changed
- src/gpu/kernels/dequant_shared.wgsl — removed dead `@compute` entry point `main_f16_out` (not pinned in digests) and its orphaned `override WORKGROUP_SIZE_F16` (only referenced by the removed entry). Re-visited despite fire-3 clean status — new finding type (dead entry point vs. earlier dead const).
- src/config/kernels/kernel-ref-digests.js — re-synced after WGSL edit
- src/tooling/litert-package-runtime.js — replaced local `cloneJsonValue` with import from `src/utils/clone-json.js`
- src/tooling/source-package-profiles.js — same
- src/tooling/source-runtime-materializer.js — same
- src/tooling/source-runtime-converter-config.js — same
- src/inference/browser-harness-text-helpers.js — same
- src/tooling/hf-registry-utils.js — removed dead export `normalizeToken` (zero external importers, zero internal callers)
- src/tooling/hf-registry-utils.d.ts — paired type removal

### Deleted
- src/tooling/bench-runner.js — entirely unused module (no importers in src/, tools/, tests/, demo/; not in package.json exports; no `.d.ts`)
- src/tooling/lora-runner.js — same: entirely unused (LoRA commands route through `loadTrainingOperatorModules()` dynamic import, not this file)
- src/tooling/distill-runner.js — same: entirely unused (distill commands route through `loadTrainingOperatorModules()`)
- src/tooling/verify-runner.js — same: entirely unused

### Visited clean (skipped from future fires)
- src/tooling/litert-package-runtime.js (consolidated — punt remaining duplicate helpers `normalizeText` local variant)
- src/tooling/source-package-profiles.js
- src/tooling/source-runtime-materializer.js
- src/tooling/source-runtime-converter-config.js
- src/tooling/source-artifact-adapter.js (verified — `normalizeText` variant B, deferred)
- src/inference/browser-harness-text-helpers.js
- src/tooling/hf-registry-utils.js (export removed)
- src/tooling/node-command-runner.js (verified — does not dispatch deleted runners, uses dynamic import for lora/distill)
- src/tooling/browser-command-runner.js
- src/tooling/command-runner-shared.js
- src/tooling/command-api.js
- src/tooling/command-envelope.js
- src/gpu/kernels/dequant_shared.wgsl (re-visited, now truly clean)
- src/inference/runtime-model.js (verified — Variant B normalizeText, deferred)

### Punts
- `normalizeText` Variant B (`String(value || '').trim()`) still duplicated across 4 production files (tooling/litert-package-runtime, tooling/source-artifact-adapter, tooling/source-package-profiles, inference/runtime-model). Consolidation would need a separate `src/utils/coerce-text.js` with distinct naming from the strict `normalizeText` in `src/utils/plain-object.js`-adjacent land. Deferred.
- Command runner dead-file sweep found 4 dead `*-runner.js` files this fire; similar pattern may exist elsewhere in `src/tooling/` (e.g. `node-convert-worker.js`, `command-api-helpers.js`, `command-validation.js`). My naive grep flagged them as no-importers but I didn't verify deeply (they may be dynamically imported). Deferred.
- `cloneJsonValue` consolidation now spans 9 total migrated sites across fires 6 + 7. Remaining potential sites in demo/, tools/, and tests/ were intentionally not touched — public-facing demos may prefer local copies for bundler isolation. Deferred.
- Pre-existing codegen patches broken for 6 variants (carried over from fire-1 onward).

## fire-6 — 2026-04-19 UTC

Landings (7+): 7
WGSL touches: 2   JS touches: 5 (incl. paired .d.ts + 2 deletions + 2 new utils)

Baseline parity vs fire-5: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* violations (unchanged), `test:unit` 26/349
(unchanged). No regressions.

### Changed
- src/gpu/kernels/softmax.wgsl — removed 2 dead `@compute` entry points `softmax_inplace` and `log_softmax` (neither pinned in `kernel-ref-digests.js`, zero JS dispatch)
- src/gpu/kernels/bf16_to_f32.wgsl — removed dead `@compute` entry point `main_single` (not pinned, not dispatched)
- src/config/kernels/kernel-ref-digests.js — re-synced after WGSL edits
- src/tooling/opfs-cache.js — replaced local `cloneJsonValue` with import from shared util
- src/converter/core.js — same
- src/converter/conversion-plan.js — same
- src/storage/source-artifact-store.js — replaced both local `cloneJsonValue` and local `encodeUtf8` with imports from shared utils
- src/tooling/source-runtime-bundle.js — replaced both local `cloneJsonValue` and local `encodeUtf8` with imports from shared utils
- src/gpu/uniform-cache.js — removed dead methods `evictStale`, `getPendingDestructionCount` (zero external refs in src/, tools/, tests/, demo/; only `.d.ts` declarations)
- src/gpu/uniform-cache.d.ts — paired type removals
- src/gpu/profiler.js — removed 3 dead exports `getProfiler`, `createProfiler`, `timeOperation` (zero external refs; only `.d.ts` declarations)
- src/gpu/profiler.d.ts — paired type removals
- src/debug/index.js — removed re-export of `perf` + import + DOPPLER_API entry + default-export entry
- src/debug/index.d.ts — paired removals

### Deleted
- src/debug/perf.js — deprecated module explicitly marked `deprecated; use performance.now() or gpu/profiler.js instead`; zero callers of `perf.mark()`/`perf.measure()`/`perf.time()` anywhere in src/, tools/, tests/, demo/
- src/debug/perf.d.ts — paired `.d.ts` for deleted module

### Added
- src/utils/clone-json.js — canonical `cloneJsonValue` (null-guard + structuredClone + JSON fallback variant; picked as strict superset of the 3 variants found across 10 sites)
- src/utils/encode-utf8.js — canonical `encodeUtf8` (static `TextEncoder` + `String(value ?? '')` coercion matching both prior local copies)

### Visited clean (skipped from future fires)
- src/utils/clone-json.js, src/utils/encode-utf8.js
- src/tooling/opfs-cache.js, src/tooling/source-runtime-bundle.js
- src/converter/core.js, src/converter/conversion-plan.js
- src/storage/source-artifact-store.js
- src/gpu/uniform-cache.js, src/gpu/uniform-cache.d.ts
- src/gpu/profiler.js, src/gpu/profiler.d.ts
- src/gpu/kernels/softmax.wgsl, src/gpu/kernels/bf16_to_f32.wgsl
- src/debug/index.js (re-touched — deprecated-perf removal justifies re-visit)
- src/debug/index.d.ts

### Punts
- `normalizeText` consolidation attempted but bailed: two semantically different variants in use (`typeof === 'string' ? trim : ''` vs `String(value || '').trim()` — diverge for number/boolean inputs). Partial consolidation would break callers that rely on coercion of non-strings. Keep split for now; if a strict-form migration is desired later, rename the stricter variant to `normalizeStringOrEmpty` to make the semantic explicit.
- `cloneJsonValue` still duplicated in 6 more sites after this fire (litert-package-runtime, source-package-profiles, source-runtime-materializer, source-runtime-converter-config, browser-harness-text-helpers, and one more). Deferred to a follow-up sweep — same DRY finding, just more migration to do.
- `isCached` method on `UniformBufferCache` appears dead externally (only internal uses) but was kept because it's part of the documented class API and test harnesses may construct instances directly. Flagged for a deeper audit fire.
- Pre-existing codegen patches broken for 6 variants (carried over).

## fire-5 — 2026-04-18 UTC

Landings (7+): 7
WGSL touches: 4   JS touches: 3 (incl. paired .d.ts)

Baseline parity vs fire-4: `kernels:check` 6 pre-existing errors (unchanged),
`imports:check:browser` 18 pre-existing node:* violations (unchanged), `test:unit` 26/349
(IMPROVED from 30/349 — likely unrelated transient fixes across the 4 prior committed
fires, not directly caused by this fire's changes). No regressions.

### Changed
- src/gpu/kernels/fused_matmul_q4_batched_f16a.wgsl — removed unused `const BLOCK_SIZE` doc-constant
- src/gpu/kernels/fused_matmul_q4_f16a.wgsl — same
- src/gpu/kernels/fused_matmul_q4_multicol_f16.wgsl — same
- src/gpu/kernels/fused_matmul_q4_multicol_f16a.wgsl — same
- src/config/kernels/kernel-ref-digests.js — re-synced after WGSL edits
- src/gpu/perf-guards.js — removed 5 dead exports: `getPerfSummary`, `logPerfSummary`, `enableProductionMode`, `enableDebugMode`, `enableBenchmarkMode`. Zero external callers (grep showed only `.d.ts` declaration + `.js` definition for each)
- src/gpu/perf-guards.d.ts — paired type removals
- src/gpu/kernel-selection-log.js — removed 2 dead exports: `resetKernelSelectionLog`, `getKernelSelectionLog`. Also dropped the now-orphan `selectionLog` array backing them; only `logKernelSelectionOnce` remains (actually used)
- src/gpu/kernel-selection-log.d.ts — paired type removals
- src/gpu/submit-tracker.js — removed 2 dead exports: `setSubmitPhase`, `estimateBatchingSavings`. Note: phase-tracking infrastructure (`currentPhase`, `phaseStats`) stays — without callers of `setSubmitPhase`, `currentPhase` remains `'other'` for all submits, but that's a latent behavior issue not introduced by this fire (grep showed zero existing callers of `setSubmitPhase` in src/, tools/, tests/)
- src/gpu/submit-tracker.d.ts — paired type removals

### Visited clean (skipped from future fires)
- src/gpu/perf-guards.js, perf-guards.d.ts
- src/gpu/kernel-selection-log.js, kernel-selection-log.d.ts
- src/gpu/submit-tracker.js, submit-tracker.d.ts
- src/gpu/kernel-selection-cache.js (verified live — `markWarmed` called from `generator.js` and `model-load.js`)
- src/gpu/tensor.js
- src/gpu/weight-buffer.js
- src/gpu/buffer-pool.js
- src/gpu/readback-utils.js (verified live via test, not a dead-module despite no src/ callers)
- tests/gpu/readback-utils.test.js (covers readback-utils.js)
- src/gpu/kernels/fused_matmul_q4_batched_f16a.wgsl
- src/gpu/kernels/fused_matmul_q4_f16a.wgsl
- src/gpu/kernels/fused_matmul_q4_multicol_f16.wgsl
- src/gpu/kernels/fused_matmul_q4_multicol_f16a.wgsl

### Punts
- `cloneJsonValue` duplicated across 10 files (storage, tooling, converter, inference). Consolidation requires a new util module + 10 call-site migrations — too large for one landing per the ≤100-line rule. Next fire can take the first 3-4 sites.
- `encodeUtf8` near-duplicate between `src/storage/source-artifact-store.js` and `src/tooling/source-runtime-bundle.js`. Trivial but partial — would need to touch 2 files in this fire; deferred to pair with cloneJsonValue consolidation batch.
- `onnxruntime-web` devDep flagged again; still possibly transitive via `@huggingface/transformers`. Needs peer-dep audit.
- Uniform-cache class methods `isCached`/`getPendingDestructionCount` flagged by agent as dead — verification would require full class-level audit. Deferred.
- `normalizeNodeBrowserCommand` flagged as dead export in `src/tooling-exports.js` — would need to verify it's not part of published API surface before removal. Deferred.
- Pre-existing codegen patches broken for 6 variants (carried over).

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
