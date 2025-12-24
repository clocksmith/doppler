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
- attention
- normalization
- ffn
- rope
- output
- layerPattern
- chatTemplate
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
| Session | runtime → default | ✗ (throw) | ✓ | n/a |
| Hybrid | call → runtime → manifest → default | ✓ | ✓ | ✓ |

### Examples

| Category | Examples |
| --- | --- |
| Generation | temperature, topK, topP, repetitionPenalty, maxTokens, stopSequences |
| Model | slidingWindow, attnLogitSoftcapping, ropeTheta, rmsNormEps, activation |
| Session | activationDtype, kvDtype, batchSize, logLevel |
| Hybrid | useChatTemplate, kernelPath |

---

## Harness Restrictions

Runtime tunables are config-only when using the browser harness:

- Command intent and harness options live in config (`runtime.shared.harness` and `runtime.shared.tooling.intent`).
- `calibrate` intent forbids tracing, profiling, probes, and debug-only benchmarks.
- Harness URLs accept only `runtimePreset`, `runtimeConfig`, `runtimeConfigUrl`, or `configChain`. No per-field URL overrides.
- Kernel selection overrides are config-only via `runtime.inference.kernelPath`.

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
   - Doppler-only: `runtime.inference.batching.batchSize`, `runtime.inference.batching.readbackInterval`, `runtime.inference.kernelPath`
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
- Loader must not mutate inference config.
- Shared runtime is the only cross-cutting config between loader and inference.
- Defaults live in schema files; runtime code should not hardcode fallbacks.
- Rule maps are config assets: JSON-only, data-only, and loaded via the rule registry.
- Runtime must not silently escalate precision to `f32`.
- Any `f32` activation path must be explicit in config/manifest and documented as a stability or capability choice.

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
