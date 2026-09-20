# Execution ownership and dependency checks

This explains the implementation boundaries; it does not replace the
[architecture](architecture.md), component charters, or release qualification.
Forge decides computation. Capsule Runtime validates and executes approved
programs. Host assembly supplies acquisition, storage, devices and model handles.

## Resource ownership

| Resource | Owner and contract |
| --- | --- |
| Pipelines, shader modules and layouts | Device-keyed caches; explicit device selection is propagated from dispatch through compilation. Layout identity is not its debug label. |
| Pooled buffers and uniforms | Their originating device/pool, including deferred release after another device becomes current. Loss invalidates only that device. |
| Loaded tensors and expert cache | A model-owned loader; injected loaders preserve their declared ownership. |
| Numerical configuration | Validated immutable pipeline snapshot. Changing compatibility defaults affects future loads, not active requests. |
| Storage | Opened backend/model handles, with pinned manifest metadata for download writes and resume reads. Close and cancellation do not retarget another handle. |
| Generation request | One immutable normalized request consumed by execution and evidence, including `presencePenalty`. |
| Sequence state | A narrow reset contract validates a non-negative safe integer before truncation or mutation. Invalid strings, fractions, NaN and negative numbers are rejected. |

The legacy pipeline still contains ambient helper APIs. Its host-created async
operations run inside the existing serialized shader-source compatibility scope,
with the owning configuration and device installed and restored on every exit.
This is an explicit compatibility boundary, **not a claim of simultaneous
independent execution by the old engine**, and not a new minimal interpreter.
Factory-created handles retain session snapshots; direct legacy singleton APIs
remain compatibility defaults. Reentrant application callbacks should not await
another operation requiring the same serialized execution scope.

Scoped-session `close()` blocks new work immediately. Concurrent close and
asynchronous-disposal callers await the same unload; later calls retain its
success or original failure without repeating cleanup. Contract interleavings
are exercised in `tests/inference/pipeline-session-interleaving.test.js` and
`tests/integration/doppler-scoped-session.test.js`; these use resource doubles,
not physical model execution.

The existing pipeline scope also owns operation exclusion. Adapter preparation
and activation are one exclusive operation; active or queued execution rejects
adapter changes and external state resets before mutation. Preparation rejects
new execution on the same owner, while other owners retain their own state.
Multiple handles for the same pipeline share this guard. Generation, embedding
and reranking retain the scope through evidence construction; their internal
pipeline calls do not acquire a second scope. Reranking's internal per-document
reset remains permitted, without allowing application callbacks to reset it.
The compatibility adapter provider uses the same owner: rejected attach/detach
operations leave its registry unchanged, and temporary adapter restoration stays
inside the operation, before unloading can begin.

Closing seals the owner synchronously, rejects queued work when it reaches the
execution boundary, and waits for active work and adapter preparation to settle
before unloading. A prepared adapter cannot activate after close or device loss.
This does not forcibly interrupt adapter I/O or submitted GPU commands. Consumers
must finish or return paused streams before awaiting their close. Preparation
failures preserve their original error, and cleanup cannot restore loader
metadata after unload. These contracts are exercised by
`tests/integration/lora-session-ownership.test.js` using non-physical fixtures.

The pipeline classes remain compatibility surfaces. Sampling normalization,
stop-sequence detection and sequence rollback have independently testable inputs;
other extracted model-step functions still retain their existing numerical
implementation. No model mathematics, WGSL, signed Capsule or historical execution
receipt is rewritten by this consolidation.

## Storage and optional transport

`openModelReadSession(modelId)` captures a model-bound read handle.
`openModelStoreSession(modelId, manifest)` adds lifecycle-bound shard and file
access for acquisition. Both are exported through the storage tooling slice.
Downloads automatically open their own store, and reject overlapping downloads
for the same model. Different models may load and download concurrently.

Storage owns HTTP acquisition, verification, retries, resume, source transitions
and delivery metrics. It does not import model preparation or peer tooling.
Hosts may pass `DownloadOptions.transport`; the optional distribution adapter is
`downloadShardWithOptionalDistribution`. Configuring peer delivery without an
explicit host transport fails. The legacy model manager supplies that adapter.
The peer implementation remains under its existing experimental owner; HTTP does
not require it. Delivery deduplication preserves storage, cancellation and peer
transport identity.

Neutral source descriptions live in `formats/source-runtime`; storage-context
construction lives in `storage/source-storage-context`. Source loading lives in
`client/model-host/source-loading`. Executable graph and capability transforms
belong to `converter`; the reusable catalog/URL-driven harness belongs to
`tooling`. Historical import paths re-export those owners without duplicating
algorithms. Compatibility facades remain visible in cycle and ownership checks.

## Mechanical boundaries

`tools/lib/module-dependencies.js` parses JavaScript and declarations for the
existing architecture checker, Capsule inventory, package closure and dependency
report. It distinguishes imports, forwarding exports, literal lazy imports,
worker URLs, runtime asset URLs and declaration imports. Computed dependencies
require a declaration or remain explicit diagnostics. Packaging conservatively
includes parsed relative asset strings without presenting them as runtime imports.

The [generated dependency views](architecture-dependencies.json) separate the
injected core from host execution and separate eager, lazy, tooling, experiment,
application and test reachability. They are potential dependencies, not measured
runtime traces. The small Capsule inventory is explicitly an **injected-core**
inventory; it cannot establish a complete application's footprint.

Commands:

```sh
npm run source:dependencies:check
node tools/report-source-dependencies.js --changed=src/storage/shard-manager.js
npm run source:architecture:check
npm run typecheck:source
```

Strict implementation checking explicitly enables `noImplicitAny` and
`strictNullChecks` for the boundary roots in `tsconfig.source-strict.json`, and
checks that sibling declarations have not hidden those JavaScript files.
`tests/types/ownership-contracts.js` checks invalid consumers as well as valid
contracts. The existing declaration-consumer check remains separate.
[Type debt](../tools/policies/source-type-debt.json) inventories unchecked
implementations and declaration `any` counts; the checker rejects growth and
ratchets reductions. This is incremental checked JavaScript, not a claim that
the whole repository is strict or a TypeScript migration.

## Integration baseline and verification

The [local acceptance snapshot](../reports/architecture-consolidation/20260919/acceptance.json)
binds the exact archive, command logs, installed consumers and scoped physical
receipts. It is not a publication or a model-support promotion.

This work starts from `a5d7a7d2` on the existing PR #10 reconciliation branch,
which includes main `086b49dd`. Its same-device repair, streaming tests and
historical physical evidence are retained rather than recreated. The stale
Glimmer preflight observation is preserved under
`reports/capsule-migration/history/` before generating a current candidate.

Boundary tests cover invalid requests without mutation, executed/evidenced
settings, out-of-order close, cancelled iteration, changing defaults, explicit
device compilation, isolated device loss and overlapping storage operations.
Installed-consumer tests and physical tests remain separate evidence classes;
neither implies the other. Rebuild the package after executable changes and bind
standalone and Reploid acceptance to the same archive. Historical signatures and
receipts retain their original identity.

`check:green` reports independent checker failures without suppressing later
correctness tests; dependencies inside an individual check still fail closed.
The architecture-debt gate remains enforced. The explicit audit authorization
for the reviewed boundary moves and exact compatibility inventories is:

```text
sha256:66797cdee28d91342930af4179ca9cc290da31cdd7634bce1786aaabe1a6fc4f
```

When comparing this change against its old policy baseline, the maintainer's
`DOPPLER_ARCHITECTURE_DEBT_AUTHORIZATION` must match that digest. No CI variable,
remote branch, publication, trust key or release claim is changed by local
validation.
