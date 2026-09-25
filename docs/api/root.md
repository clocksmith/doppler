# Capsule Runtime API

## Purpose

Start with `doppler-gpu/host` when using Doppler's existing browser or Node host
composition. It accepts a Capsule URL/path, explicit trusted signers and accepted
TargetPlan digests, and supplies the existing device, artifact-store, and program
ports. For Capsule objects, supply an artifact store. It does not choose trust or
accept upgrades. See the [application example](../../README.md#capsule-runtime-api)
and [Electron renderer](../../examples/electron-document-search/README.md).
An exported host path is not proof that every model works on that host.

`doppler-gpu` is the Capsule-native production entrypoint. It validates a signed
Doppler Capsule, selects one already-qualified TargetPlan for the observed device,
then verifies every artifact and reachable WGSL module, binds resources, and
executes the declared commands. It does not load unsigned manifests, infer a
model family, choose kernels, or supply a development signing key.

`doppler-gpu/runtime` is an exact alias of the same entrypoint.

Capsule v2 remains readable. The new `doppler-gpu/capsule` facade supplies v3 migration,
identity, and signed release-event APIs. See [Capsule identity migration](../capsule-identity-migration.md)
for explicit trust/checkpoint policy, artifact-source ownership, and sequence
execution receipts. New source APIs do not imply a published or qualified release.

## Partition-helper import correction

The corrected runtime candidate removes the accidental, undeclared partition
helper exports from `doppler-gpu` and `doppler-gpu/runtime`. These repository
helpers do not execute or qualify distributed inference and remain internal;
there is no supported public replacement operation. The minimal injected runtime
and its published declarations agree again. The immutable search starter's
pinned 0.6.2 archive is unchanged; adopting a corrected runtime requires a
separate installed-application qualification.

## Import path

```js
import {
  DOPPLER_VERSION,
  RUNTIME_CORE_VERSION,
  createDopplerRuntime,
  createFetchCapsuleArtifactStore,
  openCapsule,
} from 'doppler-gpu';
```

## Explicit-port authority

When using `doppler-gpu` or `doppler-gpu/runtime`, the application injects:

- `device`: the concrete WebGPU resource and capability adapter.
- `artifactStore`: reads and hashes Capsule artifacts.
- `trustedSigners`: the explicit signer-ID to public-JWK trust map.
- `programFactory`: loads generic execution mechanisms for the selected plan.

No port has a behavior-changing default. Missing ports fail before Capsule
validation. The compatibility facade's `modelLoadOptions` are prohibited on
Capsule execution because they could rewrite signed execution policy.

## Minimal example

```js
import { createFetchCapsuleArtifactStore, openCapsule } from 'doppler-gpu';

const capsuleUrl = new URL('./model.capsule.json', import.meta.url).href;
const capsule = await (await fetch(capsuleUrl)).json();
const artifactStore = createFetchCapsuleArtifactStore(capsuleUrl);

const session = await openCapsule(capsule, {
  device,
  artifactStore,
  trustedSigners: new Map([[signerId, signerPublicKey]]),
  programFactory,
});

const { text, tokenIds } = await session.generateText(generationOptions);
console.log(text, tokenIds, session.selectedTargetPlanDigest);
await session.close();
```

## Acquisition, cancellation, and verified reads

Opening authenticates signed metadata and release permissions before selecting
an application-approved, operation-qualified device plan. An incompatible plan
causes no artifact reads. Authenticated release denials still advance the
application's durable checkpoint; cancellation and device incompatibility cannot
erase them. Metadata-only verification is not permission to execute: the full
artifact closure must pass byte verification before constructing a program.

Session opening accepts `signal`, `loadTimeoutMs`, `maxMetadataBytes`, `maxRetainedArtifactBytes`, and
`onLoadProgress`. Pass these directly to `doppler-gpu/host` or
`runtime.openCapsule()`, and under `options.session` for the explicit-port root
`openCapsule()` facade. The JSON loading policy
defaults the deadline to `null` (no deadline) and HTTP metadata to a 16 MiB byte
limit. Set a positive `loadTimeoutMs` to bound acquisition; the deadline includes
metadata and artifact downloads. Progress events contain `phase` (`metadata` or
`artifact`), `artifactId` (null for metadata), `loadedBytes`, and `totalBytes`
(null until metadata size is known). They contain no document inputs.

```js
const controller = new AbortController();
const session = await openCapsule(capsule, {
  device, artifactStore, trustedSigners, programFactory,
  session: {
    signal: controller.signal,
    loadTimeoutMs: 120000,
    onLoadProgress: event => console.log(event),
  },
});
```

The HTTP adapter cancels response streams and checks signed artifact sizes during
download. Node file reads also receive the signal. Custom sources receive
`readArtifact(artifact, { signal, onLoadProgress, ... })` and must honor the signal
to stop their own I/O. A cancelled read is never admitted, even if a custom source
ignores cancellation. Cancellation prevents further loading reads and closes a
program that finishes construction after cancellation; it does not preempt
submitted GPU work or an uncooperative resource constructor. Opening signals and
deadline listeners are detached when opening finishes. Execution has its own
per-operation cancellation control. Electron translates opening cancellation to
its existing `DOPPLER_ELECTRON_CANCELLED` error.

Sources may additionally implement `streamArtifact(artifact, options)` returning
an async iterable of ordered, nonempty `Uint8Array` chunks. Options include
`signal`, `onLoadProgress`, and the required `maxChunkBytes` bound. Each chunk's
entire backing buffer must fit that bound; a small view of a large allocation is
not bounded acquisition. The source must honor cancellation, release on iterator
return, and keep the yielded bytes stable until the next pull. HTTP and Node file
adapters implement this port; peer hosts can supply it without importing peer
coordination into inference. A failed stream is rejected, never silently retried
through whole-file acquisition. Existing `readArtifact()` sources remain valid
but their complete source allocation is not bounded by this streaming contract.

`maxAcquisitionChunkBytes` defaults to 1 MiB and bounds one outstanding source
chunk per acquisition. `verificationYieldBytes` defaults to 1 MiB; verification
returns to host timer tasks at that byte interval, including for whole-file
sources. Timers are scheduled before the batch to overlap their delay with work;
this is not microtask-only yielding or an off-thread throughput claim. Multiple
independent acquisitions have separate workspace and cancellation. Private 64 KiB
blocks become shared backing only after the complete size and digest match.

The internal verified store owns immutable, private byte-block snapshots of
admitted artifacts. It copies and hashes at most 64 KiB per owned block; source
buffers and returned arrays cannot mutate those blocks. Range reads allocate only
their requested output; full-file reads still require space for that file.
The loading policy defaults `maxRetainedArtifactBytes` to `null` (unlimited).
A nonnegative byte limit bounds the optional full-file metadata byte cache using
least-recently-used eviction; zero disables that cache, not verified backing.
Weight shards use their backing directly, avoiding another retained JavaScript
copy alongside model resources. Eviction never triggers reacquisition or rehashing
while the store owns its immutable snapshot. A new store verifies its source
again after the final owner releases its snapshot. The limit excludes owned
backing, active reads, returned slices, loader allocations, and GPU memory.
Sessions opened by one host service share completed, verified snapshots by
content hash and exact size, while retaining independent authorization, cache,
cancellation, and cleanup. No source digest claim is trusted. Concurrent pending
acquisitions remain independent; only completed verification is shared. The
final store close releases backing references; no process-global content cache
or browser disk quota is required. Backing still occupies model-sized memory;
its logical byte count is not a physical RAM measurement.

`hashArtifact()` reports that internally computed verification without copying or
rehashing a retained file. `readArtifactRange()` returns only an owned requested slice;
neither callers nor source buffers can mutate verified backing. Externally supplied
hash claims are never trusted by Capsule opening. Manifest-level shard checks remain
separate because they bind another identity. For a factory-created verified store,
the Capsule adapter compares each SHA-256 manifest shard's normalized path, size,
and digest against the store's owned snapshot receipt. Matching identities reuse
that verification without reading and hashing the shard a second time. The store
interface is immutable, so its byte readers cannot be replaced after this check.
An arbitrary port, copied interface, or proxy does not gain this ownership guarantee
by advertising a digest method. Such ports, and manifests using BLAKE3 shard
digests, retain byte-level manifest verification. Mismatched identities fail closed.

Observer events `capsule-validation-complete` and `capsule-load-complete` include
`artifactMetrics`: source bytes read, bytes hashed by the verified store, bytes
copied there, retained and peak-retained bytes, and bytes returned to consumers.
`evictions` counts removed cached files. `sourceReadMs`, `hashingMs`, and
`copyingMs` measure successful source reads, store hashing, and owned copies;
source reads include work performed inside the supplied artifact adapter.
`backingBytes`, `peakBackingBytes`, and `backingFiles` count leased snapshots;
`snapshotCopiedBytes` counts bytes copied into newly verified backing.
`sharedBackingBytes` counts leases on existing verified backing, not new memory;
do not sum per-session leased bytes as unique allocation. `peakSnapshotBlockBytes`
measures the largest owned block, separately from fixed SHA state/schedule workspace.
`peakSourceChunkBytes` reports the largest input backing buffer (including the
whole input for legacy sources); `streamedSourceBytes` counts streamed input, and
`verificationYields` counts completed host-task yields. Source byte counters may
include rejected input; they are work counters, not proof of accepted identity.
These counters exclude HTTP internals, manifest hashing, GPU upload, driver
allocation, and total process memory. They establish copy/verification work, not
a measured latency or peak-memory improvement. Capsule acquisition does not provide
persistent browser storage or offline application-shell installation by itself.

`generationOptions` contains only SessionPlan values admitted by the Capsule,
including prompt tokens, output limit, sampling tuple, stop policy, and abort
signal. It cannot change graph topology, precision, fusion, kernel selection,
KV layout, or memory strategy.

## Generation request and completion contract

For operation events, see [incremental Capsule streams](../capsule-streaming.md).
Request v2 carries incremental partials and a complete final result; v1 retains
its cumulative event contract. The public reconstruction helper avoids cumulative
copying during partial consumption.

`doppler-gpu/generation-contract` exports the validation-only public contract:
`GENERATION_CONTRACT`, `validateGenerationInput`, `resolveGenerationOptions`,
and `GenerationError`. Its JSON definition owns accepted fields, ranges,
defaults, stopping reasons and error codes; declarations are generated from it.
It imports neither an engine nor storage. Consumers can bundle this public
entrypoint and check it against `session.generationContract` before execution.

`executeOperation()` generation input is exactly one of `{ prompt }` or
`{ promptTokens }`. Supply explicit token and sequence budgets, temperature,
top-k, top-p, repetition penalty/window, and chat-template policy. Stochastic
requests require a nonnegative seed. Presence penalty defaults to zero;
suppression and stop-sequence lists default to empty in the canonical contract.
A zero repetition window includes the whole context, and top-k zero retains
all finite candidates. Apply repetition then presence penalty once per distinct
token in the window; normalize top-k before retaining the top-p prefix.

Completed generation returns raw `text`, `tokenIds`, resolved `sampling`, and
`completion` with prompt/output token counts and the actual `stopReason`:
`max-tokens`, `eos-token`, `stop-token`, or `stop-sequence`. `generateText()` also
returns `modelId`; operation receipts bind the model identity and complete
output. Streaming iterators return completion metadata when exhausted. Cancelled
or expired operations throw `DOPPLER_GENERATION_ABORTED` or
`DOPPLER_GENERATION_DEADLINE` and do not issue successful completion receipts.
Cancellation cannot preempt GPU commands already submitted.

Migration: unknown/invalid generation settings now fail with
`DOPPLER_GENERATION_INVALID_REQUEST`; applications must consume the supported
contract rather than copy option lists. Shared CPU filtering replaces the
legacy pipeline's differing top-p order. GPU sampling now retains exact top-k
across partitions and applies top-p; F16 input storage uses declared F32
candidate/probability accumulation. Shader changes require new identity-bound
qualification, not relabeling old Capsule pins. Operator parity and injected
public-API tests alone do not qualify a model or answer. Doppler performs no
citation formatting, semantic review, passage selection, or application replay.

## Text embeddings

`session.embed()` is a Capsule-backed text operation, distinct from protein
`encodeSequence()`. It requires a passed `embed` qualification on the selected
TargetPlan and current host surface. Generation, reranking, or sequence evidence
cannot authorize it. A qualification record identifies `embeddedTexts` and its
reference transcript digest; adding this API does not qualify an embedding model.
Forge materializes `doppler.embeddingModelQualification.v1` reports into
operation-specific transcripts. Its source comparison rechecks exact input
token IDs and every vector component against the reference's explicit absolute
tolerance, binds the source revision and manifest postprocessor, and rejects
failed observations even if a report labels itself passed. Source references,
observed output vectors, and rejected comparisons must be retained separately.

```js
const result = await session.embed({
  application: acceptedRelease.application,
  text: 'A document to index locally.',
  options: { signal: abortController.signal },
});
console.log(result.embedding, result.receipt.targetPlanDigest);
```

The application identity must match the verified release (the selected release
event for Capsule v3). The signed manifest must explicitly declare embedding
support, hidden size, pooling, projection geometry, prompt inclusion, and
normalization through `output.embeddingPostprocessor`. Currently prompt
exclusion and call-time semantic overrides are rejected. The existing WGSL
pipeline performs pooling, projection, and normalization; the Capsule boundary
does not calculate or modify vectors on the CPU.

The returned embedding and token arrays are immutable snapshots. The execution
receipt binds the input and application, output, manifest identity, selected
TargetPlan, verified artifacts, backend observation, and release event where
applicable. Hash consistency establishes record binding, not numerical accuracy
or truthful remote execution. Independent references must qualify the output.
Cancellation is checked before execution and after completion and forwarded to
the pipeline; it does not promise to preempt already-submitted GPU commands.
Device loss requires closing the session and explicitly reopening it.

`executeOperation()` uses these same checked session methods, not raw program
ports. Its `embed` input is `{ texts, application }`; omitting application
authority fails before execution. Each completed item retains its embedding
receipt, and the batch receipt binds all items to the operation request and
assignment. `encodeSequence` likewise uses its qualified session method and
passes the assignment into its sequence receipt. Neither adapter may borrow
generation qualification. Partial events are not application acceptance;
cancellation or invalid evidence prevents a completed operation receipt.

## Request-bound adapters

`session.executeOperation(request, { signal, adapterArtifactStore })` accepts an
immutable `adapterSet` in the request. Each `doppler.capsule-adapter/v1` entry binds
its application-approved identity, exact base model (`modelId`, `semanticRoot`,
`envelopeDigest`, `artifactClosureDigest`), format, manifest, and weight artifact.
See the [adapter declarations](../../src/config/capsule-adapters.d.ts) for the full
request shape. The request hash and completion receipt bind the adapter set.

The application owns publication admission, distribution permission and fetching.
`adapterArtifactStore.readArtifact(artifact)` supplies the exact authorized bytes;
Runtime checks their size and SHA-256 before the existing LoRA loader activates
them. There is no adapter URL fallback or base-model duplication.

`peft_safetensors` declares PEFT matrix orientation: A is `[rank, input]` and B
is `[output, rank]`. The format resolves through
[layout policy](../../src/config/lora-layouts.json); the loader retains the
declared shapes and WGSL reads that orientation directly. Conflicting layout
or projection geometry fails before dispatch. Layout is included in the active
adapter identity. Raw model-handle imports can declare `weightsLayout: 'peft'`
in their manifest or load options. The configured legacy raw-import layout
remains `input-major` for existing Doppler training exports; those exports
must not be labeled `peft_safetensors` without conversion.

Adapter execution requires a signed TargetPlan v2 `adapterExecution` declaration:

```json
{
  "schema": "doppler.capsule-adapter-execution/v1",
  "maxAdapters": 1,
  "combination": "single",
  "formats": ["peft_safetensors"],
  "operations": ["generate"],
  "kernelModules": ["exact-matmul-module", "exact-scale-module", "exact-residual-module"]
}
```

Module IDs and digests must match both the initial execution identity and the
packaged TargetPlan closure, and include the
registered operations required by [adapter policy](../../src/config/capsule-adapters.json).
Forge takes this declaration through its existing JSON `adapterExecution` input;
it is preserved in the TargetPlan before hashing and signing. Example module IDs
above are placeholders, not a runnable model configuration. Missing declarations,
incompatible base identities, corrupted bytes and unsupported combinations fail
before model execution. Existing requests without adapters retain their declared
legacy empty set; they do not gain adapter permission.

Runtime checks the active adapter identity and bundled revocation state around
partial events and completion. Adapter receipts retain the supplied identity,
descriptor hash, source-byte digest and loaded tensor identity. Cleanup unloads
the adapter and resets generation state on completion, cancellation and failure.
These bindings do not establish specialist quality; each base/adapter combination
still needs its own retained physical and reference evidence.

## Session execution ownership

Direct generation, embedding, reranking, forecasting, sequence execution and
`executeOperation()` share one exclusive session slot. Concurrent work or reset
fails explicitly. Different session objects retain separate ownership.
`close()` requests cancellation, closes a paused stream, and waits for active work
and adapter cleanup before disposing the program and verified storage. It does
not promise immediate GPU termination. Callers must exhaust or close iterators;
partial output remains provisional until the completed record is validated.

The legacy pipeline adapter classifies methods with explicit operation contracts
(execution, streaming, mutation/reset, inspection, shutdown), never function
constructor names. Newly exposed methods require a declaration. Finish or return
paused advanced pipeline iterators before awaiting their unload operation.

Embedding batch scheduling owns only a copied prompt list, request cancellation,
and an injected embedding operation. It no longer holds ambient configuration or
shader scope across the complete batch: another session may execute between
prompts. Each numerical embedding call still uses the protected compatibility
scope. This is not a claim of simultaneous independent GPU inference.

Session unload releases its KV/decode buffers and model-owned fused weights;
shared RoPE buffers remain live until their last lease closes. Pool observations
separate retained model buffers, other active allocations, reusable idle buffers,
and deferred destruction. Those counts exclude direct allocations outside the
pool and do not measure driver residency. Closing a session never destroys a
shared device. A device-owning host may trim its pool or destroy its own device
only after its consumers finish; borrowed devices remain the caller's responsibility.

## TargetPlan v2 initialization gate

For a `doppler.target-plan/v2` target, `programFactory` must return a program
whose `getInitialExecutionIdentity()` reports the resolved graph, reachable
kernel closure, dtype lane, fusion set, KV layout, memory policy, execution-plan
digest, and runtime-engine identity. Runtime compares that canonical identity
with the signed TargetPlan before creating the resource binder or dispatching
prefill. Any mismatch fails closed.

Newly forged ModelIR v2 targets require
`doppler.initial-execution-identity/v2`. Its signed `programLoadPolicy` contains
only the fully resolved runtime `session`, `compute`, and
`generation.disableMultiTokenDecode` JSON required to create the declared
execution plan. Public `modelLoadOptions` remain prohibited. Runtime applies
this Capsule-owned policy before loading the mature execution mechanism, then
independently observes and compares the complete identity before prefill.
Program-load policy v1 remains readable for rejected or previously frozen
evidence, but Forge promotes only reconstructive policy v2. Initial execution
identity v1 remains accepted only for compatibility with already frozen Capsule v0
targets.

## Architecture boundary

JSON declares semantic and execution policy. JavaScript validates, binds, and
orchestrates. WGSL computes only declared tensor operations. Every reachable
shader is content-addressed; changing shader bytes changes TargetPlan and Capsule
identity.

## Code pointers

- [Capsule runtime entrypoint](../../src/capsule-runtime.js)
- [Runtime composition root](../../src/client/runtime/composition-root.js)
- [Initial execution identity](../../src/config/initial-execution-identity.js)
- [Compatibility API](compat.md)
