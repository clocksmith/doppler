# Config Parameter Resolution

This document defines strict parameter categories and the resolution order used
by DOPPLER. The rules are enforced at runtime with errors or warnings.

## Categories

| Category | Resolution Order | Call-time | Runtime | Manifest |
| --- | --- | --- | --- | --- |
| Generation | call → runtime → default | ✓ | ✓ | — |
| Model | runtime (experimental) → manifest → default | ✗ (throw) | ✓ (warn) | ✓ |
| Session | runtime → default | ✗ (throw) | ✓ | — |
| Hybrid | call → runtime → manifest → default | ✓ | ✓ | ✓ |

## CLI + Harness Restrictions

Runtime tunables are config-only when using the CLI or test harnesses:

- CLI flags must not override prompt, max tokens, sampling, trace, log levels, or warmup/timed runs.
- Harness URLs accept only `runtimeConfig` and optional `configChain`. No per-field URL overrides.
- Kernel selection overrides are config-only via `runtime.inference.kernelPath`.

When you need a change, create a preset or pass `--config` with a runtime config file.

## Category Examples

| Category | Examples |
| --- | --- |
| Generation | temperature, topK, topP, repetitionPenalty, maxTokens, stopSequences |
| Model | slidingWindow, attnLogitSoftcapping, ropeTheta, rmsNormEps, activation |
| Session | activationDtype, kvDtype, batchSize, logLevel |
| Hybrid | useChatTemplate, kernelPath |

## Model Params: Experimental Override

Model params are manifest-primary. Runtime overrides are supported for
experimentation and will emit a warning:

```
Experimental: Overriding 2 model param(s) via runtime: rope.ropeTheta, attention.slidingWindow.
```

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
