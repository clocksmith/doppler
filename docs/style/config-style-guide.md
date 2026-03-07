# DOPPLER Config Style Guide

Rules and conventions for configuration ownership, merge order, and overrides.

---

## Config Schema Layout

### Root Schemas

ConverterConfigSchema
- quantization
- sharding
- weightLayout
- manifest
- output
- presets

RuntimeConfigSchema
- shared
- loading
- inference

ManifestInferenceSchema (embedded in manifest.json)
- schema
- attention
- normalization
- ffn
- rope
- output
- layerPattern
- chatTemplate
- sessionDefaults
- execution
- defaultKernelPath

### Runtime Subschemas

SharedRuntimeConfigSchema (cross-cutting for loading + inference)
- debug
- tooling
- benchmark
- platform
- kernelRegistry
- kernelThresholds
- bufferPool
- gpuCache
- memory
- tuner
- hotSwap
- bridge
- harness

LoaderConfigSchema (runtime.loading)
- storage
- distribution
- shardCache
- memoryManagement
- opfsPath
- expertCache

InferenceConfigSchema (runtime.inference)
- batching
- sampling
- compute
- tokenizer
- largeWeights
- kvcache
- moe
- pipeline
- kernelPath
- chatTemplate
- prompt
- modelOverrides

### Batching Defaults (runtime.inference.batching)

- batchSize
- maxTokens
- stopCheckMode
- readbackInterval (null = read back each batch)
- ringTokens / ringStop / ringStaging (null = disable ring allocation; used for batch decode and single-token GPU sampling readback reuse)

---

## Category Rules

| Category | Resolution Order | Call-time | Runtime | Manifest |
| --- | --- | --- | --- | --- |
| Generation | call → runtime → default | ✓ | ✓ | n/a |
| Model | runtime (experimental) → manifest → default | ✗ (throw) | ✓ (warn) | ✓ |
| Session | runtime → default² | ✗ (throw) | ✓ | n/a¹ |
| Hybrid | call → runtime → manifest → default | ✓ | ✓ | ✓ |

¹ Exception: in execution-v0 manifests (`manifest.inference.schema == "doppler.execution/v0"`), `manifest.inference.sessionDefaults` is the base layer for session resolution — `resolvedSession = merge(manifest.inference.sessionDefaults, runtime.inference.session)`. The `n/a` applies only to non-execution-v0 manifests for `runtime.inference.session`.
² Separate from `runtime.inference.session`, non-execution-v0 manifests may still provide `manifest.inference.sessionDefaults.decodeLoop` as manifest batching defaults. Loader applies those defaults only while runtime batching/generation are still at global defaults; otherwise it fails closed on conflict.

### Examples

| Category | Examples |
| --- | --- |
| Generation | temperature, topK, topP, repetitionPenalty, maxTokens, stopSequences |
| Model | slidingWindow, attnLogitSoftcapping, ropeTheta, rmsNormEps, activation |
| Session | activationDtype, kvDtype, batchSize, logLevel |
| Hybrid | useChatTemplate, kernelPath |

## Fallback Rule: What Is Allowed

- No behavior-changing fallback may be implemented as ad-hoc JS branching.
- All allowed fallbacks are explicit in config/rule assets:
  - manifest and presets (for supported model/config combinations)
  - rule-map alias entries
  - capability-based kernel and dtype policy in registry overlays
- Any unresolvable choice must raise a typed error before execution begins.

---

## Harness Restrictions

Runtime tunables are config-only when using the browser harness:

- Command intent and harness options live in config (`runtime.shared.harness` and `runtime.shared.tooling.intent`).
- `calibrate` intent forbids tracing, profiling, probes, and debug-only benchmarks.
- Harness URLs accept only `runtimePreset`, `runtimeConfig`, `runtimeConfigUrl`, or `configChain`. No per-field URL overrides.
- Kernel selection overrides are config-only via `runtime.inference.kernelPath`.
- Capability remap policy is config-only via `runtime.inference.kernelPathPolicy`.
- Runtime-input precedence is normative, not surface-local:
  `configChain -> runtimePreset -> runtimeConfigUrl -> runtimeConfig -> runtime contract patch`.
- If a harness or runtime surface cannot support one of those inputs, it must fail closed instead of silently skipping it.

When you need a change, create a preset or pass a runtime config file via `runtimeConfigUrl`.

---

## Merge Order

Runtime config merge order:

```
runtimeConfig = merge(runtimeDefaults, runtimePreset, runtimeOverride)
```

Manifest inference config merge order:

```
manifestInference = merge(manifestDefaults, modelPreset, converterOverride, artifactDerived)
```

Model inference config merge order:

```
modelInference = merge(manifestInference, runtimeInferenceOverride)
```

Execution-v0 runtime compile order (when `manifest.inference.execution` exists):

```
require manifest.inference.schema == "doppler.execution/v0"
require runtime execution overlay keys == {session, executionPatch}
resolvedSession = merge(manifest.inference.sessionDefaults, runtime.inference.session)
resolvedSteps = applyPatch(manifest.inference.execution.steps, runtime.inference.executionPatch)
validate step contract: src,dst required; non-cast steps require kernel + kernelRef
validate kernelRef pinning: each step.kernelRef must resolve exactly in session.compute.kernelProfiles
resolvedPrecision = step.precision -> kernelProfile.precision -> session.compute.defaults
resolvedKVIO = step.kvIO -> kernelProfile.kvIO -> session.kvcache.kvDtype
validate precision capability: if kernel variant declares output dtype metadata, resolved outputDtype must be supported by kernel shader+entry
validate graph: no dangling slots, prefill->decode boundary compatibility, KV dtype compatibility
runtimeInference = merge(runtime.inference, compiledExecutionRuntimePatch)
```

Conversion bridge rule:

```
if inference.defaultKernelPath is present, inference.execution is absent,
and layerPattern does not require custom conv scheduling,
converter derives execution-v0 steps/session defaults from the kernel path.
If converterConfig.inference.sessionDefaults is also provided, it overlays the
generated execution-v0 session defaults before validation.
```

---

## End-to-End Config Ownership

Use explicit config domains end-to-end:

- **Conversion config**: conversion-time artifact policy only (`quantization`, sharding, output model ID/path, manifest policy).
- **Runtime config**: execution policy only (`shared`, `loading`, `inference`).
- **Benchmark shared config**: fairness/workload contract only (prompt shape, token budgets, sampling, seed, warm/cold mode).
- **Benchmark engine overlay**: engine-specific execution knobs only (Doppler batch/readback/kernel path, TJS runtime backend/session knobs).

Do not mix benchmark fairness axes with engine internals in the same config object.

---

## Benchmark Config Split (Required)

Benchmark definitions must be composed from two layers:

1. **Shared benchmark contract** (applies to all engines):
   - prefill/decode lengths
   - sampling (`temperature`, `topK`, `topP`, seed)
   - stop conditions
   - run policy (`warmupRuns`, `timedRuns`, `cacheMode`)
2. **Engine overlay**:
   - Doppler-only: `runtime.inference.batching.batchSize`, `runtime.inference.batching.readbackInterval`, `runtime.inference.kernelPath`, `runtime.inference.kernelPathPolicy`
   - TJS-only: backend/session plumbing that does not alter fairness semantics

Calibration comparisons are invalid if shared-contract fields differ across engines.

---

## Preset Registry Metadata (Required)

Runtime/config presets must carry lifecycle metadata in the preset object or registry:

- `id`
- `intent` (`verify` | `investigate` | `calibrate`)
- `stability` (`canonical` | `experimental` | `deprecated`)
- `owner`
- `createdAtUtc`
- optional: `supersedes`, `deprecatedAtUtc`, `replacementId`

Agents should resolve presets by metadata/intent, not filename heuristics.

---

## Generated Assets

- VFS manifest generation has been removed from Doppler package+demo mode.

## Rules

- Converter → manifest is the only bridge into runtime.
- `manifest.inference.schema` is required for execution-v0 manifests and must be `doppler.execution/v0`.
- execution-v0 steps require explicit `src` and `dst`.
- execution-v0 non-cast steps require explicit `kernel` and pinned `kernelRef`.
- `kernelRef` is exact-match pinned (`id`, `version`, `digest`) against `sessionDefaults.compute.kernelProfiles`.
- `kernelRef.digest` is WGSL-content pinning (`sha256(normalized shader source + entry)`), not filename-only identity.
- Loader must not mutate inference config.
- Shared runtime is the only cross-cutting config between loader and inference.
- Defaults live in schema files; runtime code should not hardcode fallbacks.
- Rule maps are config assets: JSON-only, data-only, and loaded via the rule registry.
- Runtime must not silently escalate precision to `f32`.
- Any `f32` activation path must be explicit in config/manifest and documented as a stability or capability choice.
- Execution patch semantics are atomic and ordered: `set -> remove -> add`.
- `executionPatch.set` may only edit `precision`, `kvIO`, `constants`, `entry`.
- Runtime execution overlay is strict: only `runtime.inference.session` and `runtime.inference.executionPatch` feed execution-v0 compile.

---

## Errors

Invalid usage throws immediately:

```
DopplerConfigError: "slidingWindow" is a model param. Cannot override at call-time.
Set via runtime.inference.modelOverrides (experimental) or manifest.
```

```
DopplerConfigError: "batchSize" is a session param. Cannot override at call-time.
Set via setRuntimeConfig() before generation.
```
