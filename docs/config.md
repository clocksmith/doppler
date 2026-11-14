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

### Selected Qwen expert ranges

`profiles/flash-next-routed-qualification` enables
`loading.expertCache.qwen35Routed` instead of the whole-bank pager. It requires
on-demand, immediate `topk-route` execution and untransformed row-major Q4K
expert tensors. Incompatible execution or a storage backend without bounded
range reads fails explicitly. This profile changes loading, not inference
precision or model qualification policy.

`cacheBytes` bounds an LRU cache of individual learned experts. A reusable
logical bank preserves global expert offsets for the existing kernels; its
allocation is additional to the cache. `gpuBudgetBytes` covers this owner's
buffers plus the device buffer pool, leaving `reserveBytes` for activation/KV
growth. It is a process accounting limit, not a system-wide GPU-memory cap.
Eviction waits for submitted GPU work, and cancellation drains owned reads and
cleanup before releasing the operation.

`stagingBytes` bounds source payload plus assembled ranges across
`readConcurrency` readers; backend/driver caches are separate. Prefill deduplicates
actual selections in `routeChunkTokens` chunks and retains already populated
working-bank slots within a layer. Reads/uploads for the next chunk can overlap
previously submitted computation; there is no predicted-expert prefetch.

`hostCacheBytes` optionally bounds a separate CPU cache of assembled, unchanged
expert payloads; zero disables it. `hostCacheAdmission: "all"` retains first-use
payloads, while `"repeat-read"` retains an expert only after it needs another
source read. The latter avoids retaining one-use prefill experts. The bound
includes retained payloads and concurrent admission reservations, excludes
backend/driver caches and metadata, and is separate from GPU accounting.
Cancellation discards partial payloads; unload drains readers and clears the
cache. Cache hits still upload the original packed bytes to the GPU.
`hostCacheAdmissionPhase: "all"` preserves phase-independent admission. Set it to
`"decode"` to allow prefill hits while preventing prefill misses from admitting
host entries, evicting retained decode payloads, or seeding repeat-read history.
This composes with either admission policy and requires the actual runtime phase,
including for one-token prefill. Hits still refresh recency. GPU LRU and byte
budgets are unchanged; skipped host admission does not remove transient read or
upload allocations. Phase admission/skip counters distinguish retention changes
from measured throughput improvements.

`profiles/flash-next-node-host-cache-qualification` explicitly requests a 16 GiB
host cache with repeat-read admission and the existing F32-KV profile. It is a
Node investigation profile, not a new default or a model qualification claim.

Node direct-source readers expose an explicit `readInto` capability: the tensor
reader owns its final output and the file reader fills only the supplied byte
view. This avoids an intermediate source allocation and assembly copy for
untransformed expert ranges, including multi-shard spans. Generic sources without
the capability retain read-and-copy behavior, so borrowed source buffers cannot
alias retained host cache payloads. Source verification, exact-length checks,
cancellation, staging budgets, and quantized bytes are unchanged.

With `runtime.shared.debug.profiler.enabled: true`, routed expert statistics also
include `phaseProfiles` for prefill, decode, and unspecified callers. Each bucket
reports preparation wall time, host lookup time, existing queue-fence wait time,
counter deltas (source reads, uploads, hits, evictions, submissions), and observed
GPU/host/staging peaks. Concurrent read and synchronization durations overlap:
these operation sums are **not** an additive wall-time decomposition or GPU
execution timings. No extra fence is inserted. Disabled profiling creates no
phase snapshots or new timing calls; existing counters remain available. Measure
the whole profiler toggle's overhead separately because it also enables existing
runtime profiling outside the cache.

`loader.getRoutedExpertCacheStats()` exposes source/upload bytes, cache and
working-bank hits, evictions, owned GPU bytes and staging high-water marks.
Host hits, misses, admission skips, evictions, retained/reserved bytes and their
combined high-water mark are reported separately.
`ioWaitMs` and `sourceReadWaitMs` sum reader durations, so they are not request
latency. `uploadEnqueueMs` measures CPU enqueue time, not completed transfer.
`initialQueueIdleForWeightsMs` counts only the first proven idle gap per
preparation on this client queue; it is a lower bound, not global GPU idle time.
Physical disk I/O and end-to-end latency require separate host measurements.

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

## Prefill submission bounds

`runtime.inference.session.prefillChunkLayers` controls the layers recorded in
one prefill chunk. In `prefillChunkSubmitMode: "async"`, the runtime-only
`prefillMaxInFlightChunks` limits outstanding chunks, including the chunk being
recorded. The schema default is `2`; a positive safe integer is required.
Before recording beyond this limit, Run waits for the oldest submitted chunk
and its deferred cleanup. A value of `1` waits at every boundary. Profiling and
`"sync"` submission retain their existing per-chunk waits.

This bounds queued intermediate-buffer lifetimes without changing shader math,
token chunking, or model facts. It does not bound one chunk's allocation, resident
weights, the buffer pool, or total device/process memory. Failure and cancellation
still settle submitted work before releasing its resources. The resolved runtime
config records the selected limit; manifests do not own this scheduling field.

## Warm generation at load

`runtime.shared.kernelWarmup.warmGeneration` (`null` by default) is
`{ prompt, maxTokens }`: after a text pipeline loads its weights, it runs one
greedy generation of the raw prompt (no chat template) and then resets sequence
state. The first request then finds the pipelines of the model's own plan
compiled and, with `decodeLoop.wholeTokenProgram`, a built program that its
first decode step verifies before replaying. The time is load time (phase
`warmGeneration`); `pipelineLoadTiming.details.warmGeneration` records the
tokens generated, the stop reason and whether a program is ready. A stop token
before `maxTokens` is reported, not an error; an invalid policy or a failed
generation fails the load.

## Whole-token replay windows

With `session.decodeLoop.wholeTokenProgram: true`, `batchSize > 1` and
`generation.disableMultiTokenDecode: false`, the prepared program can queue
consecutive greedy steps using GPU token feedback. `batchSize` is a ceiling:
ordinary token delivery starts the answer, then the window grows with completed
decode steps, bounded by the configured cap and declared program positions.

This transport requires contiguous whole-model state, no independent convolution
or active finiteness fallback policy, `stopCheckMode: "batch"`,
`readbackMode: "sequential"` and `readbackInterval: 1`. Each step starts its own
readback after submission. Tokens stream in order while later queued steps run;
the arithmetic and declared pipelines remain unchanged.

The window retains a recurrent/convolution checkpoint. EOS, stop sequences,
abort, consumer return and delivery errors drain queued work, then restore and
replay the published prefix when needed. Extra tentative tokens are never
published. Checkpoint, readback and recovery costs belong to request wall time.
Cancellation waits for already queued GPU work; the configured window bounds it.

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

KV-cache length contract:
- `inference.session.kvcache.maxSeqLen` has no code default. A runtime profile
  or override wins; otherwise the manifest session value (stamped from the
  conversion config) applies. The model context (`modelConfig.maxSeqLen`) caps
  the result.
- Published manifests that predate the declaration (Gemma 3/4, TranslateGemma,
  EmbeddingGemma, LFM2, DiffusionGemma) take their value from
  `src/config/runtime/model-sessions.json`, keyed by exact `modelId`, without
  changing manifest bytes. Each entry records the manifest context it was
  computed from and the length the model ran with before the code default was
  removed, `min(4096, contextLength)`; a manifest with a different
  `architecture.maxSeqLen` fails at load instead of taking a stale length. A
  manifest that declares its own value is used instead of the entry.
- When none of these declares it, KV allocation and RoPE setup fail fast with
  an error naming each place; a run never falls back to a hidden length.
  `tests/config/model-kv-declaration-contract.test.js` checks that every tracked
  manifest, catalog entry and quickstart model resolves a length.

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

## RDRR source verification buffers

`loading.shardCache.verificationBufferBytes` defaults to `0`, retaining
incremental source verification. A positive safe integer admits complete source
buffers up to that size for RDRR HTTP range loading through the shared command
harness. Larger shards still verify incrementally. Buffered verifications are
serialized within each storage context; failed checks release the queue and
remain retryable. Complete-byte SHA-256 uses WebCrypto where available, with
the same declared hash algorithm and expected digest.

The internal `createArtifactStorageContext` and
`createHttpArtifactStorageContext` constructors accept the same
`verificationBufferBytes` option explicitly. Direct-source containers reject a
positive budget because their large-file verification remains incremental. This is loading policy,
not a manifest inference field, precision change, or permission to skip checks.
The budget bounds the verifier's source buffer; native crypto may copy its
input. Full-shard consumers already require a full buffer and verify that exact
returned buffer regardless of this optional range-verification budget.

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

### Hashed n-gram table storage

`inference.ngramEmbedding.tableDtype` declares retained `bf16` or `q4_0` rows.
Q4_0 uses 32 values per 18-byte block; `headDim` must be divisible by 32.
For example, a 160-value row occupies 90 bytes, not 320. Row caches and bounded
range reads account for stored bytes. Tensor dtype, shape and physical byte
length must agree with the declared storage format before binding. The source
converter may supply `ngram_table_dtype` independently of the model's other
weight precisions. This does not qualify a mixed-precision model end to end.
