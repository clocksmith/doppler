# Doppler Config Style Guide

Rules and conventions for configuration ownership, merge order, and overrides.

## Config Schema Layout

### Root Schemas

- `ConverterConfigSchema`: quantization, sharding, weightLayout, manifest, output
- `RuntimeConfigSchema`: shared, loading, inference
- `ManifestInferenceSchema` (embedded in manifest.json): schema, attention, normalization, ffn, rope, output, layerPattern, chatTemplate, session, execution

### Runtime Subschemas

- `SharedRuntimeConfigSchema` (cross-cutting for loading + inference): debug, tooling, benchmark, platform, kernelRegistry, kernelThresholds, bufferPool, gpuCache, memory, tuner, hotSwap, bridge, harness
- `LoaderConfigSchema` (runtime.loading): storage, distribution, shardCache, memoryManagement, opfsPath, expertCache
- `InferenceConfigSchema` (runtime.inference): batching, sampling, compute, tokenizer, largeWeights, kvcache, moe, pipeline, kernelPath, chatTemplate, prompt, modelOverrides

### Session Decode Loop (runtime.inference.session.decodeLoop)

- batchSize, maxTokens, stopCheckMode
- readbackInterval (null = read back each batch)
- ringTokens / ringStop / ringStaging (null = disable ring allocation; used for batch decode and single-token GPU sampling readback reuse)

Recorded decode submits at `batchSize` token boundaries; `readbackInterval`
groups those batches before token readback but never merges them into one
unbounded command buffer. Resources stay owned until the grouped work completes,
including aborts after a submitted prefix. Execution statistics count actual
submissions; submit latency covers the final submission through completion.
Earlier submissions may execute during host recording; full decode wall time
includes both, without subtracting overlapped recording twice.

## Category Rules

| Category | Resolution Order | Call-time | Runtime | Manifest | Examples |
| --- | --- | --- | --- | --- | --- |
| Generation | call → runtime | ✓ | ✓ | n/a | temperature, topK, topP, repetitionPenalty, maxTokens, stopSequences |
| Model | manifest; allowlisted multimodal runtime overlay only | ✗ (throw) | allowlist only | ✓ | slidingWindow, attnLogitSoftcapping, ropeTheta, rmsNormEps, activation |
| Session | runtime → manifest | ✗ (throw) | ✓ | n/a¹ | activationDtype, kvDtype, batchSize, logLevel |
| Hybrid | call → runtime → manifest | ✓ | ✓ | ✓ | useChatTemplate, kernelPath |

¹ `resolvedSession = merge(manifest.inference.session, runtime.inference.session)`: the manifest owns the baseline (including the decode loop); runtime session overrides win field-by-field.

## Fallback Rule: What Is Allowed

- No behavior-changing fallback may be implemented as ad-hoc JS branching.
- Runtime defaults for tunables live in `src/config/schema/*.schema.js` and
  checked-in runtime profiles/config assets. Manifest defaults live in the
  conversion config and stamped manifest. Run code consumes the resolved
  object only.
- Direct constructors, browser helpers, and execution-plan builders must reject
  missing resolved config fields, never recreate schema defaults locally.
- All allowed fallbacks are explicit in config/rule assets: manifest and
  conversion/runtime config assets (for supported model/config combinations),
  rule-map alias entries, and capability-based kernel and dtype policy in
  registry overlays.
- Capability transform policy lives in
  `src/rules/inference/capability-transforms.rules.json`. Every rule carries
  `kind`, `dtypeEffect` (classifies any dtype change), and `evidence`. When you
  edit these rules, run `npm run capability-policy:check`.
- Any unresolvable choice must raise a typed error before execution begins.
- `npm run config:single-source:check` inventories known fallback-drift
  patterns. Run it when you edit a file it scans; add a pattern only for a
  fixed drift likely to recur.

### Capability Transform Categories

| Kind | Purpose | Identity Rules |
| --- | --- | --- |
| `hardware-compatibility` | Missing hardware feature adaptation (`hasF16=false`, `hasSubgroups=false`) | no model/platform match |
| `runtime-session-compatibility` | Resolved session compatibility such as `kvDtype=f32` | no model/platform match |
| `explicit-lane` | Requested f16 or selective-f16 runtime lane | exact model/list or generic `requiresF16ActivationNarrowing=true` |
| `platform-workaround` | Known platform bug workaround | exact platform plus exact model/list and evidence |
| `lane-mismatch-guard` | Fail closed on manifest/profile lane mismatch | exact model/list |
| `capability-optimization` | Positive capability optimization without dtype change | exact model/list and positive capability |
| `default-noop` | Final no-transform rule | empty match, last rule |

An explicit f16 lane is not a fallback. A hardware fallback never matches model
identity.

## Harness Restrictions

Runtime tunables are config-only in the browser harness:

- Command intent and harness selection stay in command/suite context, not `runtime.shared.*`.
- `calibrate` intent forbids tracing, profiling, probes, and debug-only benchmarks.
- Harness URLs accept only `runtimeProfile`, `runtimeConfig`, `runtimeConfigUrl`, or `configChain`; no per-field URL overrides.
- Kernel selection and capability remap policy are config-only via `runtime.inference.kernelPath` and `runtime.inference.kernelPathPolicy`.
- Runtime-input precedence `configChain -> runtimeProfile -> runtimeConfigUrl -> runtimeConfig`
  is normative on every surface; a surface that cannot support an input fails closed, never skips it.

For a change, create a profile or pass a runtime config file via `runtimeConfigUrl`.

## Merge Order

```
runtimeConfig = merge(runtimeDefaultConfig, configChainRuntimeConfig,
  runtimeProfileConfig, runtimeConfigUrl, runtimeOverrideConfig)
manifestInference = merge(manifestDefaults, converterConfig.inference, artifactDerived)
modelInference = merge(manifestInference, runtimeInferenceOverride)
```

Execution runtime compile order (when `manifest.inference.execution` exists):
expand compact tuples into resolved steps, build the inline kernel path and
layer pipeline from them, then merge `manifest.inference.session` into
`runtime.inference.session`.

`execution.inlineKernelPath: false` marks a manifest that owns an execution
graph but must not synthesize `runtime.inference.kernelPath` from it.

Conversion config rule: config must provide an explicit execution graph with
kernels, decode, prefill, preLayer, postLayer, and policies. No family
detection, no defaultKernelPath, no auto-generation. The config's execution
graph is stamped directly into the manifest.

## End-to-End Config Ownership

- **Conversion config**: conversion-time artifact policy only (`quantization`, sharding, output model ID/path, manifest policy).
- **Runtime config**: execution policy only (`shared`, `loading`, `inference`).
- **Benchmark config**: shared contract plus engine overlay, never mixed (see below).

## Benchmark Config Split (Required)

1. **Shared benchmark contract** (all engines): prompt shape, prefill/decode
   lengths, sampling (`temperature`, `topK`, `topP`, seed), stop conditions, run
   policy (`warmupRuns`, `timedRuns`, `cacheMode`, warm/cold).
2. **Engine overlay**:
   - Doppler-only: `runtime.inference.session.decodeLoop.batchSize`, `runtime.inference.session.decodeLoop.readbackInterval`, `runtime.inference.kernelPath`, `runtime.inference.kernelPathPolicy`
   - TJS-only: backend/session plumbing that does not alter fairness semantics

Calibration comparisons are invalid if shared-contract fields differ across engines.

## Profile Registry Metadata (Required)

Runtime profiles must carry lifecycle metadata in the profile object or registry:

- `id`, `owner`, `createdAtUtc`
- `intent` (`verify` | `investigate` | `calibrate`) as the profile's primary classification
- `compatibleIntents` (non-empty subset of `verify` | `investigate` | `calibrate`)
- `stability` (`canonical` | `experimental` | `deprecated`)
- optional: `supersedes`, `deprecatedAtUtc`, `replacementId`

The normalized command request owns the active intent; profile metadata
classifies and constrains a profile but cannot select it. Resolve profiles by
metadata and `compatibleIntents` (inspect with
`node src/cli/doppler-cli.js profiles --json`), not filename heuristics. CLI
`--runtime-profile <id>` only sets `request.runtimeProfile`; it never adds
per-field runtime tuning outside the config payload.

## Generated Assets

- VFS manifest generation is removed from Doppler package+demo mode.

## Rules

- Owner map (conversion, runtime, kernels, model registry, hosted HF entries,
  evidence policy): `docs/developer-guides/config-source-of-truth.md`.
- Converter → manifest is the only bridge into runtime.
- Execution steps require explicit `src` and `dst`; non-cast steps also require `kernel` and a pinned `kernelRef`.
- `kernelRef` is exact-match pinned (`id`, `version`, `digest`) against `session.compute.kernelProfiles`; `digest` pins WGSL content (`sha256(normalized shader source + entry)`), not filename identity.
- `src/config/kernels/registry.json` is the source of truth for kernel operation IDs, variant IDs, WGSL filenames, entry points, feature requirements, bindings, uniforms, and metadata.
- Uniform layouts declare each field's WGSL name, scalar type, and byte offset,
  plus the WGSL struct size. Fields are required unless marked `padding: true`;
  fields WGSL reads cannot be padding. Generated JavaScript writers reject
  missing values, out-of-range integers, and non-finite or unrepresentable floats
  before allocating a uniform buffer. Shared request objects may carry extra
  fields the selected variant ignores.
- After editing registry uniforms or their WGSL declarations, run
  `npm run kernels:uniforms:sync` (generates writers and declarations after
  checking parsed WGSL); `npm run kernels:uniforms:check` rejects declaration
  drift and stale writers. Shader compilation and physical parameter readback are
  separate checks.
- Variant `bindings` declares a complete layout, including removed or moved
  resources, and cannot coexist with the legacy partial `bindingsOverride`.
  `name` is the wrapper resource role; optional `wgslName` binds it to the shader
  variable checked by the interface generator. Matmul's
  `variantMetadata.dispatchGeometry` selects its dispatch algorithm explicitly;
  variant names never select geometry.
- Kernel-registry `weightDtype` is the source of truth for a selected kernel's
  weight encoding. Loader materialization resolves the selected inline path's
  shader and entry point against the registry, never a duplicated filename list
  or unselected execution-closure members.
- When you add, rename, or remove kernel files or registry entries, run `npm run kernels:registry:check` (`tools/sync-kernel-reachability.js --check`). Docs must not define independent kernel filename suffix grammars.
- Loader must not mutate inference config; shared runtime is the only cross-cutting config between loader and inference.
- Rule maps (including capability rules) are JSON-only, data-only config assets
  loaded via the rule registry. They must not pattern-match model identity with
  `contains`, `startsWith`, or `endsWith`. If a model-specific workaround is
  unavoidable, match an exact manifest field value or list explicit variant IDs
  with the evidence in the rule name/test.
- Run must not silently escalate precision to `f32`; any `f32` activation path
  must be explicit in config/manifest and documented as a stability or
  capability choice.
- Runtime execution overlay is strict: only `runtime.inference.session` feeds execution compile.

## Errors

Invalid usage throws immediately, e.g.:

```
DopplerConfigError: "slidingWindow" is a model param. Cannot override at call-time.
Set in the conversion config/manifest. Runtime modelOverrides are limited to an explicit allowlist.

DopplerConfigError: "batchSize" is a session param. Cannot override at call-time.
Set via setRuntimeConfig() before generation.
```
