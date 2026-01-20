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
| Generation | call → runtime → default | ✓ | ✓ | — |
| Model | runtime (experimental) → manifest → default | ✗ (throw) | ✓ (warn) | ✓ |
| Session | runtime → default | ✗ (throw) | ✓ | — |
| Hybrid | call → runtime → manifest → default | ✓ | ✓ | ✓ |

### Examples

| Category | Examples |
| --- | --- |
| Generation | temperature, topK, topP, repetitionPenalty, maxTokens, stopSequences |
| Model | slidingWindow, attnLogitSoftcapping, ropeTheta, rmsNormEps, activation |
| Session | activationDtype, kvDtype, batchSize, logLevel |
| Hybrid | useChatTemplate, kernelPath |

---

## CLI + Harness Restrictions

Runtime tunables are config-only when using the CLI or test harnesses:

- CLI accepts only config-loader flags (`--config`, `--help`).
- Command, suite, model id, and harness options live in config (`cli.*`, top-level `model`). Model is required for all CLI runs.
- CLI runs require `runtime.shared.tooling.intent` (verify/investigate/calibrate).
- `calibrate` intent forbids tracing, profiling, probes, and debug-only benchmarks.
- CLI flags must not override prompt, max tokens, sampling, trace, log levels, or warmup/timed runs.
- Harness URLs accept only `runtimeConfig` and optional `configChain`. No per-field URL overrides.
- Kernel selection overrides are config-only via `runtime.inference.kernelPath`.

When you need a change, create a preset or pass `--config` with a runtime config file.

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

## Rules

- Converter → manifest is the only bridge into runtime.
- Loader must not mutate inference config.
- Shared runtime is the only cross-cutting config between loader and inference.
- Defaults live in schema files; runtime code should not hardcode fallbacks.
- Rule maps are config assets: JSON-only, data-only, and loaded via the rule registry.
- Production inference must not use F32 weights or activations; F32 is debug-only for validation.

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
