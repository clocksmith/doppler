# Doppler Config

Runtime-facing config contract for inference and command execution.

Implementation internals for the config subsystem live in
[../src/config/GUIDE.md](../src/config/GUIDE.md).

## Canonical ownership

- Conversion vs runtime field ownership:
  [conversion-runtime-contract.md](conversion-runtime-contract.md)
- Command/surface behavior contract:
  [style/command-interface-design-guide.md](style/command-interface-design-guide.md)
- Merge and schema conventions:
  [style/config-style-guide.md](style/config-style-guide.md)

## Runtime config model

`manifest.inference.output.reasoningOutput` is an optional, model-owned output
contract. Absent or `null` preserves ordinary decoding. The explicit
`doppler.reasoning-output/v1` value contains `openingToken` and `closingToken`,
each with an integer `id` and literal `text`. Conversion validates its shape;
opening checks both markers against the actual tokenizer and rejects overlap
with stop tokens. Runtime overrides cannot change this signed model fact.

With this contract, text generation, model handles, Capsule operations and
resident partitions expose final-answer text after the naturally generated
closing marker. Raw token APIs and receipts still retain every generated token.
Resident character allocations count the full raw decoded UTF-16 stream,
including hidden reasoning and special tokens; per-step `metrics.outputUsage`
reports raw and final counts. Stop sequences apply to final text.
Natural EOS before the required boundary or an empty final answer is a typed
`DOPPLER_GENERATION_OUTPUT_INVALID` failure. Token-budget and stop-sequence
exhaustion retain their actual stop reason and a `truncated` output state;
cancellation discards decoder state without manufacturing a malformed-output
failure. Truncation is not a complete answer or task-quality acceptance.
No currently pinned model or standard archive enables this optional contract.

Doppler resolves runtime behavior from explicit runtime inputs composed in order
(each later input overrides the previous):

1. **Runtime profile** — named profile via `request.runtimeProfile` (e.g. `profiles/production`)
2. **Runtime config URL** — remote/local JSON via `request.runtimeConfigUrl`
3. **Explicit runtime override** — inline object via `request.runtimeConfig`

`configChain` is an ordered list of runtime config refs that composes ahead of
the three inputs above on surfaces that support it.

Command metadata is not a runtime-config layer. `command`, `workload`,
`modelId`, and command intent stay in the explicit command/suite request and
must not be injected into `runtime.shared.*` by the runners.

CLI flag inputs:

- `--config` accepts inline JSON, a file path, or URL for all commands.
- `profiles --json` lists checked-in runtime profile IDs and metadata.
- `--runtime-profile <id>` is an additive CLI shorthand for `request.runtimeProfile`.
  For IDs under `src/config/runtime/profiles/`, the `profiles/` prefix may be
  omitted and is restored before command normalization.
- `--runtime-config` is supported by harnessed commands (`bench`, `debug`, `diagnose`, `verify`)
  and accepts the same input shapes (`JSON`, `path`, `URL`).
- `--runtime-profile` and `--runtime-config` are mutually exclusive, and neither
  may be combined with runtime inputs already present inside `--config`.

See `src/tooling/command-runtime-execution.js:applyRuntimeInputs()` for the
merge implementation.

Behavior-changing execution choices must be present in the manifest or explicit
runtime config before execution. No hidden runtime fallbacks or command-owned
runtime rewrites are allowed.

This config boundary is the enforcement layer for Doppler's second main goal:
owning the model artifact and runtime contract. Manifest/config/schema/rule
assets may provide defaults. Run code must not invent behavior from model
names, missing fields, surface identity, or local cache state.

## MoE expert preparation

`runtime.loading.expertPreparation.mode` controls when expert banks are materialized:

```json
{ "loading": { "expertPreparation": { "mode": "eager-resident" } } }
```

The schema default, `on-demand`, preserves deferred loading. Its loader timing
receipt reports `expertPreparation.status: "deferred"`; successful opening does
not claim that every expert is resident. `eager-resident` prepares Qwen3.5-style
and Gemma4 packed banks for every assigned decoder layer before the loader emits
`complete` or the text pipeline becomes loaded. Each bank is acquired once,
without selecting or changing any learned experts. Other expert formats reject
this mode before materialization rather than silently degrading to lazy loading.

`loader.getLoadTiming()` records the selected mode, status, assigned layer count,
prepared bank count and `phasesMs.expertPreparation`. Dense models report no expert
preparation. Pipeline construction accepts `contexts.signal`; direct loader calls
accept `LoadOptions.signal`. Resident preparation waits for queued GPU uploads
before readiness. Cancellation waits for the current awaited operation and queued
uploads to settle before cleanup, prevents readiness, and preserves the failed timing
receipt for inspection. It does not promise interruption of submitted GPU work.

Eager preparation moves acquisition into opening; it does not establish faster
compute, sub-second first-token latency, bounded packed-bank eviction or a physical
GPU-memory limit. Those need separate policies and evidence. Loading configuration
does not change model weights, quantization or kernel selection.

## Multimodal config contract

Multimodal encoder behavior is config-owned, not runtime-inferred.

- `config.vision_config.vision_architecture` is required for image/video-capable models.
- `config.audio_config.audio_architecture` is required for audio-capable models.
- Vision preprocessing fields such as normalization, pixel limits, patch sizing,
  and merge sizing must be present explicitly in config or explicit runtime
  model overrides.
- Runtime does not infer multimodal architecture from `model_type`, mirrored
  top-level manifest fields, or missing tunables.

If a multimodal manifest is missing those fields, the runtime must fail fast and
the converter/config must be fixed instead of patching around the artifact.

## Immediate kernel completion

Kernel registry variants may declare `immediateCompletion: "settled"`. The
immediate executor then awaits GPU completion before returning, retaining its
uniforms through success or rejection. Abort observed during submission is
reported after settlement. `"submitted"` preserves the submission-only return
boundary; legacy variants that omit the field resolve to that value in the
registry schema. Other values, including null, reject during registry creation.
This field participates in immutable registry/execution identity.

Paired numerical variants explicitly settle immediate dispatches to bound
accumulated queued work. Recorded execution retains its recorder-owned
submission and settlement lifecycle. The policy changes neither shader math
nor reference tolerances; changed packages require their own physical
qualification. A single oversized dispatch still requires its own bound.

## Matmul workgroup geometry

Opt-in matmul variants may declare `variantMetadata.workgroupConstants` as
three constant names or `null` axes. A named axis uses the final resolved
pipeline constant; `null` retains that static registry workgroup axis. For
example, `["TILE_SIZE", "TILE_SIZE", null]` derives both tiled dimensions from
the same shader constant. Registry `wgslOverrides` retain precedence over
conflicting caller or kernel-path constants. Changing a registry override
therefore changes both the compiled workgroup and its dispatch calculation.

These variants also declare `workgroupStorageBytesPerInvocation`. Capacity
checks add that dynamic storage to the resolved static `sharedMemory` bytes.
Missing or invalid geometry/storage declarations reject before dispatch.
Current device axis, invocation and workgroup-storage limits are checked even
when the immutable geometry plan is cached. Variants without this metadata
retain their existing specialization behavior.

## Kernel path contract

Kernel paths are explicit execution plans materialized as inline objects from
the manifest execution graph. String registry IDs were removed; runtime config
must not select a kernel path by ID.

Canonical execution identity:
- conversion config `inference.execution`
- stamped manifest `inference.execution`
- execution-v1 compile output `runtime.inference.kernelPath`

Kernel-path resolution precedence is canonical in
[`conversion-runtime-contract.md`](conversion-runtime-contract.md).
This file only relies on that contract:
- `kernelPath` remains the only supported inline kernel-selection override surface
- internal runner-owned per-run context may still apply the highest-precedence
  override after manifest and runtime config resolution

`kernelPath` is the only supported kernel-selection override surface, but it
must be an inline object or `null`. Do not use legacy `kernelPlan` or removed
string registry IDs.

Kernel-path dtype contract:
- inline paths must match the resolved canonical
  `inference.session.compute.defaults.activationDtype`, output dtype, and
  `inference.session.kvcache.kvDtype`; shader identity never selects runtime dtype
- canonical activation declarations are validated and projected through the
  session runtime builder into `inference.compute.activationDtype` for legacy
  execution consumers; legacy-only declarations remain compatible
- contradictory canonical/legacy declarations and explicit runtime overrides
  fail closed, including when the resolved path would otherwise match; the
  projection does not mutate the caller or rewrite dtype to fit a shader

## Runtime boundaries

Runtime config may tune execution policy (for example batching, sampling,
intent, diagnostics), but may not rewrite conversion-owned storage facts
(quantization layout, emitted tensor set, shard hashing policy).

If a change requires different storage artifacts, reconvert the model.

Runtime overlays are allowed only for runtime-owned policy. A field that changes
artifact bytes, shard identity, tokenizer identity, quantization layout, or
emitted tensor shape belongs to conversion, catalog migration, or manifest
migration rather than a per-run override.

For investigate/verify text-generation runs, `runtime.shared.tooling.diagnostics="always"`
enables operator-timeline capture without switching the harness mode to
`diagnose`. The automatic policy is intended for compare-like debug lanes: it
emits low-overhead operator records while preserving the resolved execution
plan and batching behavior unless the caller explicitly requests a heavier
diagnostics payload.

## Command/runtime boundary

Harnessed command runs (`verify`, `debug`, `bench`) must preserve explicit
command context:
- `command`
- `workload`
- `modelId` when the workload requires one
- normalized command intent

That context belongs to the command request / suite options, not to the runtime
config merge chain. Workload-specific model ID enforcement lives in
`src/tooling/command-api-family-normalizers.js`.

## Common usage

Set runtime profile only:

```bash
node src/cli/doppler-cli.js verify --config '{"request":{"workload":"inference","modelId":"gemma-3-270m-it-q4k-ehf16-af32","runtimeProfile":"profiles/production"},"run":{"surface":"auto"}}' --json
```

Discover runtime profiles:

```bash
node src/cli/doppler-cli.js profiles --json
```

Set runtime profile with the CLI shorthand:

```bash
node src/cli/doppler-cli.js verify \
  --config '{"request":{"workload":"inference","modelId":"gemma-3-270m-it-q4k-ehf16-af32"},"run":{"surface":"auto"}}' \
  --runtime-profile profiles/production \
  --json
```

Run the experimental Gemma 4 31B all-f16 profile:

```bash
node src/cli/doppler-cli.js debug \
  --config '{"request":{"workload":"inference","modelId":"gemma-4-31b-it-text-q4k-ehf16-af16"},"run":{"surface":"browser","browser":{"channel":"chrome","headless":true,"console":true}}}' \
  --runtime-profile profiles/gemma4-31b-f16-activations-probe \
  --json
```

This profile is an investigation lane. It requests f16 activation, math,
accumulation, output, and KV cache dtypes through the Doppler-only af16 manifest
variant, which shares the af32 Q4K weight capsule through `weightsRef`.

Override runtime policy explicitly:

```bash
node src/cli/doppler-cli.js debug \
  --config '{"request":{"modelId":"gemma-3-270m-it-q4k-ehf16-af32"},"run":{"surface":"auto"}}' \
  --runtime-config '{"inference":{"generation":{"maxTokens":8}}}' \
  --json
```

Load a request from a URL:

```bash
node src/cli/doppler-cli.js verify --config https://example.org/doppler/verify-gemma.json --json
```

## Change checklist

When adding new runtime-visible behavior:

1. update schema/validation (`src/config/schema/*`)
2. update merge/loader wiring
3. update rule assets or registry data when selection behavior changes
4. add regression coverage under `tests/config` and/or integration tests
5. update this file and the relevant style guide section

## Related

- [conversion-runtime-contract.md](conversion-runtime-contract.md)
- [rdrr-format.md](rdrr-format.md)
- [operations.md](operations.md)
- [testing.md](testing.md)
