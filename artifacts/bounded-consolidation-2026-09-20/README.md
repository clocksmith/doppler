# Bounded consolidation acceptance

This is the first independently reviewable change in the requested consolidation,
not a claim that GPU attribution, operation contracts, cleanup, or model work is
finished. No publication or release promotion is performed.

The sections below retain the sequence of independently qualified candidates;
the final operation-contract candidate is described at the end. Earlier package
identities and observations are not rewritten to represent later implementations.

## Reproduced problem and repair

At unchanged `1ab605f2`, a timer-triggered cancellation of a 16 MiB verification
could not run before the entire synchronous copy/hash completed: 309.071 ms total,
297.975 ms hashing, all 16,777,216 bytes admitted. The new regression repeats that
failure mode with 2 MiB for both readers and requires rejection
before complete verification and without publishing backing.

Capable sources now implement optional `streamArtifact(artifact, options)`;
`readArtifact()` remains supported. HTTP uses a bounded BYOB reader when available;
Node files reuse one bounded read buffer. The private snapshot still owns 64 KiB
blocks. Only those owned bytes are hashed, and complete size/digest verification
precedes shared publication. Verification yields through timer tasks, not merely
microtasks. Transport remains outside `formats/sha256.js`, which is unchanged.

The default source-buffer bound is 1 MiB per active streamed acquisition. This is
not a bound on total verified backing, simultaneous acquisitions, browser network
internals, returned slices, or model/GPU memory. Legacy whole-artifact sources
still allocate their complete response. Peer hosts may implement the same port;
no peer orchestration enters inference or hashing.

Tests cover reused/mutated source buffers, arbitrary boundaries, corruption,
truncation, oversize/backing-view attacks, source failure/retry, timer cancellation,
close before publication, and independent overlapping session cancellation.
Artifact-reader declarations are canonical; HTTP and Node implementations join
the checked-JavaScript roots (9 roots, 1,047 unchecked modules).

## Frozen packages and physical scope

- Unchanged baseline: `c3166b1a6a480f52dd0661b67b427f6918eb004b015da84d31bbc86ccb57f13c`.
- Rejected first candidate: `9e9564a42fa87dc34820dedb02c843f09d50aa5222d92f3abb1ca6b33ecf3be4`.
  Real Chrome returned a default-reader buffer larger than the declared bound.
  Its failure receipt is retained; the bound was not increased to hide it.
- Corrected candidate: `58a38aa3eed85a62682ec0a6c2232217c896a7a76393335f6f3a73d46f560de1`,
  `/var/tmp/doppler-consolidation-stream-v2-20260920/doppler-gpu-0.6.2.tgz`.

The fresh baseline and corrected candidate use identical generation Capsule,
configuration, and frozen 107-token reference on Chrome/AMD RDNA 3. Both complete
the four-request, two-session lifecycle. Complete experiment elapsed times are
231,405 and 227,220 ms respectively; these are not single-request startup or TTFT.
Initial hashing is 120,707.5 versus 120,677.1 ms: no hash-speed improvement is
established. Candidate acquisition reports a 1,048,576-byte peak source chunk,
65,536-byte peak backing block, and 7,677 verification yields. Second-session
source reads, hashing, and new backing remain zero.

`hash-throughput.json` independently measures hashing 16 MiB of byte value 19
and checks against Node `createHash('sha256')`: one warmup and three timed runs.
Other checks were active, so it is not a controlled optimization comparison.
Reproduce by calling `sha256BytesHex(input)` four times around `performance.now()`;
allocate the input and calculate the independent reference before timing.

Run physical cases with:

```sh
node tools/check-installed-capabilities.js artifacts/bounded-consolidation-2026-09-20/loading-baseline-config.json
node tools/check-installed-capabilities.js artifacts/bounded-consolidation-2026-09-20/loading-bounded-config.json
node tools/check-installed-capabilities.js artifacts/bounded-consolidation-2026-09-20/loading-bounded-4gib-config.json
node tools/check-installed-capabilities.js artifacts/bounded-consolidation-2026-09-20/loading-bounded-embedding-reranking-config.json
node tools/check-installed-capabilities.js artifacts/bounded-consolidation-2026-09-20/loading-bounded-adapter-config.json
node tools/check-installed-capabilities.js artifacts/bounded-consolidation-2026-09-20/loading-bounded-reploid-v1-config.json
```

Each summary binds its original raw receipt hash and exact archive. Raw receipts
and archives remain at their recorded local paths. Existing numerical references
are unchanged; zero-delta adapter acceptance is not learned-adapter qualification.

The first Reploid run reused a v2 request fixture, but current Reploid main
`67a64fb16661eaa0a8ef34314f3fea824af5be94` declares v1 requests. Its rejection is
retained in `loading-reploid-request-mismatch-summary.json`. The corrected config
explicitly uses that supported v1 public contract, without changing model input,
reference, package, or Reploid code.

The separate `hash-state-read-*-probe` experiments compare just typed-array
destructuring with eight direct state reads. Both agree with Node's independent
SHA-256. These are diagnostic probes, not changes in the loading candidate above.
Reproduce from the repository root, passing the retained baseline implementation:

```sh
node artifacts/bounded-consolidation-2026-09-20/hash-state-read-probe.js /var/tmp/doppler-consolidation-stream-v2-20260920/consumer/node_modules/doppler-gpu/src/formats/sha256.js
node artifacts/bounded-consolidation-2026-09-20/hash-state-read-browser-probe.js /var/tmp/doppler-consolidation-stream-v2-20260920/consumer/node_modules/doppler-gpu/src/formats/sha256.js
```

## Package and boundary effects

`loading-package-audit.json` records the exact changed package entries and the
Node/npm toolchain used for the package-budget check. An unused tooling declaration
forwarder leaves the generated package closure after its consumers switch to the
canonical config declaration. Its repository compatibility file is retained.
The generated demo inventory changes; UI behavior is preserved.

Component: `doppler.runtime-source.client`, `doppler.runtime-source.config`,
`doppler.runtime-source.tooling`. Intent: preserved. Boundary effects: additive
artifact-reader contract and loading policy; no computation, shader, trust,
pooling, or execution-selection changes. Acceptance commands and outcomes are
recorded alongside this file.

## Isolated hash throughput improvement

The next candidate changes only `src/formats/sha256.js` inside the installed
archive: eight direct typed-array state reads replace destructuring before each
compression block. Round computation, incremental ownership, API, and results
are unchanged. The retained independent Node and Chrome probes above isolate
this change and check every result against Node crypto.

Archive `26f459dcaef8301601f15495df3b4103f9f3f1bb32cbd32bc7538cfb82d17616`
is retained at `/var/tmp/doppler-consolidation-hash-20260920/doppler-gpu-0.6.2.tgz`.
`hash-package-audit.json` records the installed file comparison. Generation,
4 GiB control, embedding/reranking, zero-delta adapter, and current Reploid
acceptance all pass against this same archive and unchanged numerical references.
Run each `hash-*-config.json` with `node tools/check-installed-capabilities.js`.

The generation experiment falls from 227,220 to 168,483 ms; initial hashing falls
from 120,677.1 to 61,639.3 ms. These remain complete lifecycle and verification
measurements, not TTFT claims. Bounded acquisition and second-session reuse remain
intact. This qualifies the hash change, not the later GPU cleanup changes.

Focused SHA-256 tests cover empty/padding/chunk boundaries, offset views, a
537,919,488-byte input, and independent crypto agreement. Streaming cancellation,
installed public exports/types, architecture, dependency, style, package-budget,
and closure checks pass. The preceding loading commit's full 854-file green run
is retained separately; it is not represented as a rerun on this hash candidate.

The GPU observation fixture gained allocation labels after the generation run;
later receipts record its different hash. Labels identify allocation sites, not
current ownership. They do not change the installed runtime or mathematical
references. Physical driver residency and garbage collection remain unmeasured.

Component: `doppler.runtime-source.formats`. Intent: preserved. Boundary effects:
none. No shader, model computation, release promotion, or npm publication change.

## GPU ownership repair

The hash candidate's final generation observation retained 4,949,428,824 bytes
across 486 GPU buffers: 2,264,924,160 fused QKV bytes, 2,415,919,104 KV bytes,
268,435,456 RoPE bytes, decode buffers, and small device caches. Allocation labels
were checked against their allocation and release sites; they are not residency.

The cleanup archive is
`e3a23607790a01339fd4472f07540aa7fc49dde2968d46946e47c2199d106d41`, retained at
`/var/tmp/doppler-consolidation-cleanup-20260920/doppler-gpu-0.6.2.tgz`.
Unload now destroys session KV and decode resources, registers fused QKV buffers
with the existing model loader, and releases reference-counted RoPE leases through
their originating pools. Closing one session preserves other leases. Partial
allocation and preparation failures release acquired buffers; loader cleanup
failure does not prevent session or storage cleanup. Retention policy is unchanged.

The original four-request sequence and two extra reopen/run/close cycles all
match the frozen 107-token reference. Final observed buffers plateau at 10,840
bytes across 258 objects: cached uniforms plus the two four-byte attention
fallback buffers. The intermediate first-reopen close observation still saw
8,443,421,784 bytes awaiting asynchronous reclamation; the subsequent cycle and
final snapshot return to exactly 10,840. Session close is not a GPU-idle barrier.
No synchronous GC or physical driver-residency claim is made.

The same cleanup archive passes the separate 4 GiB, embedding/reranking,
zero-delta adapter, and current Reploid cases. Generation now includes extra
cycles, so its 359,418 ms elapsed time is not comparable to the prior four-request
experiment. Raw receipts and all checkpoints remain bound by the summaries.

Focused ownership regressions, device-loss/deferred completion tests, installed
exports/types/synthetic consumers, architecture and source-style checks passed.
The source type inventory removes two unchecked declaration `any` occurrences.
No new policy exception or pool retention limit was needed; redundant blank lines
were normalized without splitting private pool state.

Component: `doppler.runtime-source.inference.pipelines.text`,
`doppler.runtime-source.memory`. Intent: preserved. Boundary effects: originating
resource ownership and additive pool allocation observations; no tensor or shader
changes. A host owns device shutdown; session unload never destroys its device.

## Explicit operations and bounded dependency cleanup

The final candidate archive is
`6471f5acbe00fa89435bc12016cd149969cdb19f78d2ef856c9245b11d9d9bf2`, retained at
`/var/tmp/doppler-consolidation-contracts-20260920/doppler-gpu-0.6.2.tgz`.
The installed source comparison checks all 1,824 shipped source entries against
the working candidate. `contracts-package-audit.json` records the separate CI
toolchain package-budget measurement; it is not the identity of another physical
test archive.

Pipeline methods now declare execution, streaming, mutation, inspection, reset,
or shutdown behavior. Function syntax and `constructor.name` no longer determine
ownership. Existing owners, aliases, adapter exclusion, late-activation rejection,
restoration, iterator cleanup, and repeated-close behavior remain in place.
Coverage checks require exposed family methods to declare their behavior. Tests
include ordinary functions returning promises and wrapped iterators.

Embedding batch scheduling has explicit prompt/request/execution ports and no
ambient configuration or shader state. Its session lease spans the batch, while
each legacy embedding call still acquires the compatibility scope. A deterministic
two-session test observes A1, B, A2 with correct per-session settings and restored
ambient state. This permits interleaving between prompts, not independent or
simultaneous GPU computation. Serialization remains around dependent compute.

Resolving selected computed imports exposed obsolete energy/diffusion module
paths and a dispatcher-to-derived-class-to-base-class cycle. Rule paths now point
to existing entrypoints; the unchanged text base classes live below dispatch.
Public text imports remain forwarding-compatible. Unused execution imports were
removed; numerical algorithms and shader bytes were not changed. The strict
source set grows to 11 roots without expanding the 1,047-module unchecked debt.

Cleanup dispositions live in the existing source-architecture inventory. Training,
distribution, energy, hotswap, and the source-loading compatibility entry are
retained with actual consumer evidence and migration requirements. No experiment
was deleted merely because it appeared test-only. The graph still reports 85
computed/unresolved diagnostics and is not claimed complete. In particular, the
training-suite computed edge exposes an undeclared quarantine bridge; resolving
its acquisition/assembly dependency needs a separate bounded repair, not a broad
exception or an assertion that the implementation is dead.

Final generation acceptance includes the original sequence and two reopen cycles.
All six outputs match the frozen 107-token reference. After queue completion and
a host-task turn, every cycle retains exactly 10,840 bytes in 258 observed GPU
objects, all attributed to small device caches. Queue completion reports no
errors. This adds an idle observation without changing session close into a GPU
barrier, forcing garbage collection, trimming pools, or claiming driver residency.

Run each `contracts-*-config.json` using
`node tools/check-installed-capabilities.js`. Summaries preserve the exact archive,
original receipt hash, fixture identities, unchanged numerical comparisons, and
measurement limits. Generation, the separate 4 GiB control, embedding/reranking,
zero-delta adapter, and Reploid all pass on that same archive. The adapter fixture
remains zero-delta lifecycle acceptance,
not qualification of a learned adapter. Publication and model promotion remain
explicit decisions.

`contracts-checks.json` records the successful full `npm run check:green` rerun
(856 test files plus independent gates), installed public export/type smoke,
demo contract check, and focused boundary checks. The earlier run also passed
856 files but failed the stale package budget; its log identity is retained.
The measured CI-toolchain budget was corrected and the complete chain rerun.

Component: `doppler.runtime-source.inference.pipelines.text`,
`doppler.runtime-source.experimental`, `doppler.repository-tooling`.
Intent: preserved. Boundary effects: explicit operation contracts, independent
batch scheduling, and base-class dependency direction; compatible public exports.

## Useful-model follow-up intake (not qualified)

The retained WGSL-repair seed29 adapter is a concrete nonzero candidate:
`artifacts/wgsl-repair/v12/external20/seed29/checkpoint-001200.adapters.safetensors`,
232,808,939 bytes, SHA-256
`0d9ab8e1348a3fcdd6ef2973b624cc1a3d962de90a571976a5a222df27b69394`.
Independent streamed Node crypto agrees with the catalog identity. The artifact
contains 256 F32 tensors and nonzero sampled data; that establishes neither
quality nor installed-runtime parity.

Its runtime manifest pins `Qwen/Qwen3.5-9B` revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` and the
`qwen-3-5-9b-f16-af32` base. That base is absent from the inspected local HF/RDRR
roots. Historical portability evidence is retained, not relabeled as qualification
of this archive. Acquiring that pinned base or selecting a supplied approved
candidate is the next model-work decision. No model download, retraining,
catalog promotion, or learned-adapter acceptance is represented as completed here.
