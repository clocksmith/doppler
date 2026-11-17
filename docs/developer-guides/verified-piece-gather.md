# Node verified-piece gathering

`src/storage/verified-piece-gather.js` provides an opt-in Node worker adapter for
compact routed experts. It is not a new qualification or trust policy. The host
supplies the pinned index identity and approved local filenames; the worker
validates the index-to-manifest binding and hashes every complete piece before
publishing a gathered range. Runtime metadata is snapshotted before admission.

Successful owned leases include frozen optional `timings` scalars on their existing
worker reply: `workerWallMs`, `readWaitMs`, `verificationMs`, `copyMs`, and lane-local
`roundTripMs`. This adds two worker clock samples and two lane clock samples per
job, validation, and fixed-size scalar metadata; it adds no IPC messages, payload
copies, or scratch buffers. Failed jobs publish no successful timing metadata.
Legacy ports may omit it; malformed present metadata rejects before publication.

Routed source counters expose `sourceTimingJobs`, `sourceUntimedJobs`, and sums
named `sourceJobWallMs`, `sourceJobReadWaitMs`, `sourceJobVerificationMs`,
`sourceJobCopyMs`, and `sourceJobRoundTripMs`. Existing profiler phase snapshots
separate prefill and decode without per-token observation calls or full receipts.
Interpret timings only with their coverage counts: unsupported jobs are not
measured zeros. Async read wait includes native I/O and callback/scheduling delay;
read waits overlap within a job, and different worker jobs overlap across lanes.
These sums are not additive components of phase wall time. Round trip is measured
entirely on the lane clock, without subtracting timestamps from different workers.

Create the adapter with `createVerifiedPieceGather({ manifestBytes, indexBytes,
indexIdentity, files, maxPartBytes, memoryBudgetBytes, concurrency })`. `files` maps authenticated
index filenames to approved local paths. Set `maxPartBytes` to cover a plane
when whole-plane gathering is desired. A larger plane is read in bounded parts.
Pass canonical regular-file paths; the worker rejects leaf symlinks and
non-regular files before reading. Nonblocking descriptor admission prevents a
replaced FIFO from stranding shutdown. An admitted descriptor owns its original
regular file until worker retirement; every acquired piece is still hashed.
Platforms without no-follow and nonblocking file-open flags reject this adapter's
file acquisition explicitly.

Production hosts using `createVerifiedPieceStorage` can supply `nodeGather`:

```js
const source = await createVerifiedPieceStorage({
  manifestBytes, indexBytes, indexIdentity, acquire, signal,
  nodeGather: { files, maxPartBytes, memoryBudgetBytes, concurrency },
});
```

`source.storage` owns both readers and closes them together. Text pipeline weight
initialization forwards its owned-range port automatically. `acquire` continues
to serve generic loads, auxiliary artifacts and unsupported tensor layouts. This
option is Node-only and requires explicit approved local files; it must not be
used to bypass custody, remote acquisition or source authorization.

`source.getReceipt()` describes only generic reads. Await
`source.getGatherReceipt()` for worker authentication evidence while no lease is
outstanding, before closing storage. These counts describe authentication work,
not successfully delivered request bytes: they may include verified prefixes of
failed reads and work completed before cancellation prevents publication. Do not
report the generic receipt as a total for both readers.

When a requested range exactly covers one complete authenticated piece, the
worker reads that piece directly into its reusable owned output and hashes the
full output with Node SHA-256 before transferring it. It resolves the range
against the common reader's validated index; lookup alone is not authentication
evidence. This avoids the generic piece scratch and intersection copy for that
case. Partial and multi-piece requests resolve pinned complete descriptors through
the common index, then serialize reads into one owned piece scratch, verify with
synchronous Node SHA-256 and copy only authenticated intersections. The worker
does not route that fallback through the browser reader's clone/WebCrypto path.
Neither path skips complete-piece authentication.
`directPieceReads` and `directPieceBytes` count successful exact-piece worker
reads, including work whose publication is later prevented by cancellation.
`copyMs` measures explicit generic gathering only, not native digest or I/O
internals. All timing fields remain cumulative worker time, not request latency.

Lower-level hosts can pass `rangeLease` through custom shard loader options:

```js
loader.setCustomShardLoader(loadShard, {
  loadShardRange,
  loadShardRangeInto,
  rangeLease: gather.rangeLease,
});
```

The ordinary shard/range callbacks still own other loads and unsupported source
transforms or multispan tensors. Compact decoding uses verified owned bytes
directly for GPU uploads instead of copying them into main-thread scratch.
Reading, hashing and gathering happen inside the worker; non-exact gathering
still copies intersections. No shared scratch backing buffer is transferred.

The optional explicit `concurrency` declares worker lanes; omission preserves
one lane. Each lane permits one outstanding read or lease with its own buffer.
Hosts may also pass `fileIdentities`, a snapshot for every approved canonical
file, with decimal-string `device`, `inode` and `size` values. This opt-in checks
the no-follow descriptor and canonical pathname before and after acquisition.
It rejects replacement even when the new file contains identical bytes; it does
not replace complete-piece SHA verification. Resolve approved input symlinks
before taking this snapshot. `rangeLease.fileIdentitiesBound` confirms this
worker capability, allowing diagnostic adapters to fail closed on older code.
Compact loading dispatches up to the smaller of this count and `readConcurrency`.
With explicit `sourceReadMode: "lease-parallel"`, compact decode dispatches
expert-plane tasks rather than assigning a lane to a whole expert. A small miss
set can therefore use the existing lanes concurrently. The default plane order
and the worker byte reservation remain unchanged; this mode requires unchanged
single-span source planes and never adds a main-thread expert scratch payload.
Saturation rejects another read rather than allocating beyond the admitted bound. It releases each lease in `finally` after
`queue.writeBuffer` has copied the bytes, including upload failures. Released
views are detached. Repeated release is harmless. Each lane buffer is reused, not
reallocated for each range. Direct callers must release a lane before exceeding the declared count.

`rangeLease.readMany(ranges, signal)` optionally gathers positive ranges up to the
explicit `maxRanges` bound (three by default, at most 32)
`{ shard, offset, length }` ranges into one concatenated owned result, in the
caller's order without padding. Their **combined** length must fit `maxPartBytes`;
it is not a separate allowance per range. Whole pieces are read and hashed into
disjoint output views in the worker. A later failure zeroes the entire result
before rejection. Release detaches all sibling views together, after every
consumer has copied its bytes. The same startup/lifetime signals and approved
file-identity checks apply to each range. Missing capability fails explicit
`sourceReadMode: "expert-lease-batch"` admission rather than changing execution.
Batch receipt counters distinguish requests, constituent ranges and payload
bytes, so an actual workload can verify the intended reduction in transfers.
For a compact-cache miss set whose entire plane count fits in the admitted
worker count, the loader can acquire one plane per worker. Each expert retains
its successful leases until all its planes authenticate before upload. Slots
and the source map publish only after every expert settles. Corruption or
cancellation settles all reads and releases every acquired lease before failure.
Larger miss sets keep bounded whole-expert groups; retained mapped-cache hooks
keep their whole-expert admission and copy behavior.
When every constituent range is a complete piece, positioned reads run
concurrently into disjoint views of that one output. A single-piece job reads
directly on its owning worker, avoiding a thread-pool handoff when there is no
sibling read to overlap. Cancellation and close still settle that read before
retirement; they do not promise an immediate interruption. Each completed read is
hashed synchronously on the owning worker, without a WebCrypto input snapshot
or a shared thread-pool digest job. Other admitted workers remain independent.
Any partial range retains the serialized generic path.
All I/O and hash jobs settle before error zeroing or transfer. Close rejects host
admission immediately, then waits for the worker to settle those jobs, zero any
unpublished output and close its descriptors before acknowledging retirement.
Thread termination follows that acknowledgement; a missing acknowledgement fails
retirement rather than proving settled I/O. Published leases remain caller-owned.

The admitted byte bound is `concurrency * (maxPartBytes + 2 * maxPieceBytes)`:
every worker's maximum output and a conservative two-piece allowance are reserved.
The generic path owns one complete-piece scratch; direct whole-piece hashing
creates no additional input copies. The existing reservation remains unchanged.
Only fixed native crypto state, immutable metadata and VM overhead are excluded;
it is not an RSS limit. Compact staging admission reserves that bound plus its
fallback scratch. Do not share the adapter between concurrently preparing caches. Multiple lanes
remove an enforced serialization point; they do not establish a measured speedup.
`activeDigestCopyBytes` and `peakDigestCopyBytes` remain in receipts for compatibility
and are zero for worker-owned synchronous hashing. Other merged peaks sum the worker
peaks as a conservative upper bound, rather than asserting simultaneous residency.
`activeGenericScratchBytes`/`peakGenericScratchBytes` expose the fallback's one-piece
scratch ownership; active ownership returns to zero before output publication or
rejection. Scratch becomes unreachable at the request boundary; this remains an
ownership bound rather than a promise of immediate VM/RSS reclamation.

For explicit boundary attribution, `model.advanced.getExpertLoadingStats()` and
the optional resident `getExpertLoadingStats()` return detached CPU snapshots of
the owning loader's cache and routing-readback counters. Ordinary `getStats()` and
generation evidence do not capture these snapshots automatically. Call at an idle
loaded-model boundary, outside a stopped request clock. Existing profiler-enabled
phase buckets distinguish prefill and decode; wait sums may overlap and never
represent GPU execution time. A profiled diagnostic needs its own source/config
identity and does not inherit qualification from an unprofiled timing run.

Cancellation prevents publication and is checked after the bounded worker read;
it does not interrupt a synchronous disk read or native hash. `gather.close()`
terminates the worker and rejects outstanding reads. The host owns closing it,
after consumers finish or cancel. Closing one adapter does not close another.
Close stops admission permanently and can be retried after a retirement failure.
The closed pool drops its buffer reference. Already published leases remain
caller-owned until release; a late release detaches that lease's own views and
does not return storage to the closed pool.

CPU tests cover authentication, short reads, transfers, aggregate admission,
upload failure, retry, cancellation and independent owners. They do not qualify
model correctness, GPU behavior or performance against other engines. No speedup
is claimed without a matched complete-request comparison.
