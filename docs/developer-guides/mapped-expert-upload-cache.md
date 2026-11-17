# Bounded mapped expert upload cache

`MappedExpertUploadCache` owns immutable GPUBuffer upload sources. It does not
read files, authenticate bytes, select model identities, or change inference
arithmetic. The caller must use keys bound to the authenticated model/index and
expert layout and supply complete authenticated bytes. No default runtime path
enables this module.

The constructor requires `device`, `maxBytes`, `maxMappedBytes`, and `maxEntries`.
`maxBytes` is a share of the existing total model GPUBuffer cache budget, not
additional memory. It charges every published, unpublished, mapped and pending
retirement buffer. Optional synchronous `allocate`/`dispose` ports integrate the
same allocations with the owning loader's accounting. Allocators must return the
exact requested size and usage; disposers must throw rather than claim success
when destruction fails. Do not create an unaccounted second heap cache.

`maxMappedBytes` must be explicitly reserved from the existing aggregate 64 MiB
transient staging allowance, alongside authenticated readers/gather outputs and
all other staging. A mapping is additionally charged there while its writer is
active; unmap or successful destruction releases that transient charge. The
same allocation remains charged to `maxBytes` until destruction. The module
allocates no heap payload copy: the callback receives the owned mapping. Source
leases held by the callback still belong to the caller's staging accounting.

- `admit(key, bytes, fill, { pinned, signal })` returns false when bounded admission
  cannot fit. `fill(view, signal)` must return the exact completed length; its
  borrowed view becomes invalid at unmap. Publication occurs only after the
  callback settles, cancellation is checked, and unmap succeeds. An existing key
  must keep its byte extent. The caller must never reuse an identity key for
  different bytes. Pinned entries remain until close.
- `copyInto(key, ranges, signal)` returns null on a miss. On a hit, it snapshots
  one to six aligned source/destination ranges and owns their encoder and queue
  submission. The source stays pinned until its actual queue completion fence;
  the returned ticket exposes that completion. It does not reread or rehash.
  Destinations are borrowed and remain caller-owned.
- Idle non-pinned entries use deterministic global LRU. Busy sources and active
  writers cannot be evicted. An admission that cannot fit does not discard other
  useful entries. Queued copies may finish after caller cancellation and retain
  their source until completion.
- `settle()` explicitly retries failed fences. A failed fence retains the source
  pin unless device loss is confirmed. `close()` stops admission immediately,
  cancels writers, waits for their callbacks and submitted copies, and disposes
  owned buffers. It coalesces concurrent closes. Failed retirement remains charged
  and can be retried with another close. It never claims a still-running writer
  or an unconfirmed native queue has settled.

`getStats()` exposes allocated/mapped peaks, published/pinned/evictable/unpublished
bytes, active writers, pending copies, entries and admission/copy/eviction counts.
These are owned GPUBuffer bytes; they are not driver residency or system RSS.

CPU verification:

```sh
node tools/run-node-tests.js tests/loader/mapped-expert-upload-cache.test.js
```

In a coordinated clean physical GPU window only:

```sh
node tests/loader/mapped-expert-upload-cache-physical.test.js
```

The short physical recipe uses synthetic 4 KiB payloads, an 8 KiB persistent
cache limit, a 4 KiB mapped-writer allowance and an 8 MiB device API ceiling.
It checks exact multi-range copies, provider-free hits, pinned retention, LRU
eviction, close with a submitted copy, caller destination ownership and zero
allocated bytes before device destruction. It is not model/source authentication
or complete-request correctness/performance qualification.

## Explicit tiered expert placement

`runtime.loading.expertCache.qwen35Routed.storageMode: "tiered-compact"`
selects `TieredExpertCache`. It requires an explicit `tiered` object containing
`residentLayers`, `parkedLayers` (disjoint original model layer indices), and
`mappedCacheBytes`. No model or placement defaults are inferred.

The planner admits complete packed source extents, device buffer/binding limits,
minimum top-K capacity and all assigned partition layers before allocating.
`cacheBytes` includes full resident banks, full parked banks, the bounded
shared mapped upload cache and every compact plane/source-index map. These
are shares of the same `gpuBudgetBytes - reserveBytes` allowance. Assigned
partitions retain original layer indices; absent resident/parked layers consume
no bank allocation. An explicit `compactLayerBytes` must cover exactly the
assigned nonresident layers and fit the remaining budget.

Setup preloads resident and parked banks through authenticated whole-expert
leases, within the public setup clock. A resident bank becomes available only
after all setup reads/uploads settle. Resident layers execute with original
expert IDs, avoiding CPU route readback. Parked experts remain immutable upload
sources; streamed layers retain bounded GPU slots during both prefill and
decode. Compact chunk sizes must fit a complete top-K selection, and kernels
receive an explicit source-ID-to-slot map. Numerical kernels and source packed
representations are unchanged.

This mode requires `sourceReadMode: "expert-lease-batch"`,
`prefillUploadMode: "expert-copy"`, `compactPrefillBytes: 0`,
`compactPrefillReuse: "none"`, `prefillCompactAdmission: "none"`, LRU and no
separate host heap cache. The range port's entire `memoryBytes` reservation plus
all concurrent mapped writers must fit `stagingBytes`. A cache instance is bound
to its admitted plan objects and source context; a changed source requires a new
loader. Failed setup requires unload before retry. Cancellation stops admission,
drains reads and submitted copies, and retains failed retirement for retry.

Focused component checks include:

```sh
node tools/run-node-tests.js tests/loader/tiered-expert-plan.test.js tests/loader/tiered-expert-cache.test.js
```

Component checks do not qualify model tokens, physical device budgets, complete
request speed or transport performance. Enable this opt-in mode in a pinned
comparison profile only after those gates pass; old storage modes remain the
default and retain their existing behavior.
