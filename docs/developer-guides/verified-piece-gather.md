# Node verified-piece gathering

`src/storage/verified-piece-gather.js` provides an opt-in Node worker adapter for
compact routed experts. It is not a new qualification or trust policy. The host
supplies the pinned index identity and approved local filenames; the worker
validates the index-to-manifest binding and hashes every complete piece before
publishing a gathered range. Runtime metadata is snapshotted before admission.

Create the adapter with `createVerifiedPieceGather({ manifestBytes, indexBytes,
indexIdentity, files, maxPartBytes, memoryBudgetBytes })`. `files` maps authenticated
index filenames to approved local paths. Set `maxPartBytes` to cover a plane
when whole-plane gathering is desired. A larger plane is read in bounded parts.

Production hosts using `createVerifiedPieceStorage` can supply `nodeGather`:

```js
const source = await createVerifiedPieceStorage({
  manifestBytes, indexBytes, indexIdentity, acquire, signal,
  nodeGather: { files, maxPartBytes, memoryBudgetBytes },
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
Reading, hashing and gathering happen inside the worker; gathering itself still
copies bytes. No shared scratch backing buffer is transferred.

One adapter permits one outstanding read or lease. Compact loading uses one
reader while this port is enabled. It releases each lease in `finally` after
`queue.writeBuffer` has copied the bytes, including upload failures. Released
views are detached. Repeated release is harmless. The buffer is reused, not
reallocated for each range. Direct callers must release before their next read.

The admitted byte bound is one maximum output plus two complete-piece byte
buffers. It excludes immutable metadata, native digest internals and VM overhead;
it is not an RSS limit. Compact staging admission reserves that bound plus its
fallback scratch. Do not share the adapter between concurrently preparing caches.

Cancellation prevents publication and is checked after the bounded worker read;
it does not interrupt a synchronous disk read or native hash. `gather.close()`
terminates the worker and rejects outstanding reads. The host owns closing it,
after consumers finish or cancel. Closing one adapter does not close another.

CPU tests cover authentication, short reads, transfers, aggregate admission,
upload failure, retry, cancellation and independent owners. They do not qualify
model correctness, GPU behavior or performance against other engines. No speedup
is claimed without a matched complete-request comparison.
