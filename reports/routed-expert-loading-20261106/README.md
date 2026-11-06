# Selected-expert Q4K loading — 2026-11-06

Component: `doppler.runtime-source.loader`, `doppler.runtime-source.inference`,
`doppler.runtime-source.config`, `doppler.tests`, `doppler.docs`.
Intent: preserved. Boundary effects: explicit opt-in loading policy; no artifact,
kernel, precision, routing, correctness-policy or serving change.

The routed loader replaces whole-bank acquisition with exact selected-expert
Q4K ranges. This is a loading optimization, not Flash-Next qualification: its
existing long-context independent-reference gate remains unchanged and failing.

## Mechanism and bounds

- Read actual GPU expert IDs; retain their identities, routing weights and the
  existing route/scatter kernels and accumulation order. Packed bytes include
  scales, block metadata and padding **per row**, including partial down rows.
- Keep a 2 GiB per-expert LRU. One reusable 1.40625 GiB logical working bank
  preserves global expert offsets without loading its unselected slots. GPU
  copies from cached payloads into this bank are not host uploads.
- The profile budgets 24 GiB of process-accounted GPU allocations, including the
  device pool, and reserves 4 GiB for activation/KV growth. This is not a physical
  or system-wide VRAM cap. Eviction fences submitted GPU work; active copy leases
  cannot be evicted. Unload drains cancelled reads and retains failed cleanup
  ownership for retry.
- Two readers share a 16 MiB payload/assembly budget. Backend/driver allocations
  are separate. Prefill deduplicates actual selections in 256-token chunks;
  already populated slots survive within a layer. Known next-chunk reads/uploads
  can overlap submitted computation. There is no speculative expert prefetch or
  unmeasured read-coalescing policy.

Use `profiles/flash-next-routed-qualification`. It inherits the existing paged
profile's inference settings and only changes loading. Unsupported recorded,
grouped/shared-slot, non-Q4K or unbounded-range execution fails explicitly. The
old pager remains available as the comparison control.

## Physical parity and lifecycle

Host: AMD Ryzen AI Max+ 395 / Radeon 8060S, Linux, Node 22.22.1, Dawn `webgpu`
0.4.0. [Hardware context](routed-hardware-context-20261106.json) records the kernel,
PCI identity, unchanged 65,945,407,488-byte GTT limit and TTM setting. The existing
27B provider stayed active with PID 3399279 before and after every measured run.
No participant service or kernel limit was changed.

The [final verification](routed-final-verification-20261106.json) checks actual
hash-verified tensor bytes, shape and finiteness:

- Long prefill: 2,304 tokens, all 5,898,240 layer-1 MoE input values and all
  5,898,240 output values are bit-exact against the old pager. All 23,040 expert
  IDs and 23,040 routing weights are bit-exact. This capture includes bounded
  chunk execution in two learned MoE layers, not a full long-model gate.
- Short generation: two requests, three full 248,320-logit rows each; all
  1,489,920 values and generated IDs are bit-exact. The prompt is
  `The capital of France is`; both modes return the same three IDs / `Paris. The`
  under the explicit three-token cap. This is parity, not an answer-quality score.
- Physical lifecycle: queued abort, active-decode rollback of committed states,
  reset/reuse, paused-consumer abort without pulling again, and paused-consumer
  unload pass. No uncaptured GPU or cleanup errors.

Long prefill fetched 960 unique experts across two layers (480 per layer),
2,831,155,200 packed bytes, with 5,288 within-layer working hits. It did **not**
exhibit decode's ten-expert traffic estimate: long prompts can request most of a
bank. [Long-prefill receipt](routed-long-prefill-parity-20261106.json).

## Controlled cost observation

[Comparison](routed-cost-comparison-20261106.json): bank and routed modes used the
same artifact manifest (`4902a0b7…ceec09`), prompt, full-logit observation, fixed
sampling, runtime source snapshot and unchanged F16 KV / F32 arithmetic. The
measured source is retained in `routed-measured-source-20261106.tar.gz`.
The explicit HC/router qualification candidate is inherited from the earlier
trace run in both modes; this is not a promotion of that candidate to defaults.

Each mode loads once, then runs two requests. “Empty” means the GPU expert cache
starts empty; “retained” resets generation state but keeps that cache. OS caches
were not cleared or controlled. Both modes ran beside the existing 27B provider.
These are individual shared-host diagnostic observations, not a population
benchmark, serving-throughput result or cross-host claim.

| Mode / cache | Prefill s | Decode 1 / 2 s | Request s | Storage API GB | Host upload GB | Physical disk GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bank / empty | 62.474 | 68.933 / 74.061 | 205.471 | 293.687 | 217.558 | 32.106 |
| Bank / retained | 81.560 | 71.636 / 71.271 | 224.469 | 293.687 | 217.433 | 48.588 |
| Routed / empty | 9.455 | 1.561 / 2.142 | 13.163 | 6.880 | 7.006 | 5.246 |
| Routed / retained | 5.414 | 1.165 / 1.037 | 7.618 | 6.736 | 6.736 | 1.327 |

GB above means decimal bytes. Storage API bytes are observed custom-reader
bytes; physical reads are `/proc/self/io.read_bytes` deltas and include OS
page-cache effects. Host uploads count `queue.writeBuffer` bytes, including small
non-expert writes; GPU-to-GPU copies and driver allocations are excluded. The
bank performs whole-shard over-read: each forward acquires about 97.9 GB through
the storage API while uploading about 72.5 GB. Routed decode reads/uploads
1.256 GB and 1.038 GB for these two selections, after cache reuse.

The empty/retained routed requests record 182/231 cache hits and 2,333/2,284
misses. Peak owned expert+working memory is 3.406 GiB; sampled pool+owned GPU
accounting peaks at 13.450 GiB; owned staging peaks at 7,372,800 bytes. Pool peaks
are separately recorded (bank 12.857 GiB, routed 10.044 GiB). Process maximum RSS
is 2.149 GiB for bank and 0.830–0.853 GiB for routed. These accounting quantities
are not additive physical-memory measurements on an integrated GPU.

The measured first client-queue idle gaps sum to 268/239 ms for the routed
requests. This is only a proven **lower bound**, not total/global GPU idle time;
we do not attribute device-global utilization while another provider remains
active. Reader-duration sums and upload-enqueue time are in the raw receipts;
neither is GPU execution/transfer time. Retained-request improvement also includes
shader and OS caches and cannot all be attributed to expert-cache hits.

## Reproduction, corrections and acceptance

The primary run JSONs and logs are retained as `routed-cost-{bank,routed}*.gz`.
On the provisioned data volume, replay each separately:

```sh
DOPPLER_ROUTED_EXPERT_COST=/path/to/run.json node tests/integration/routed-expert-cost-physical.test.js --require-model
python3 reports/routed-expert-loading-20261106/compare.py bank.json routed.json comparison.json
python3 reports/routed-expert-loading-20261106/verify-final.py /path/to/qualification-data
```

The initial optional tensor-name counter matched only GGUF aliases, so its zero
`expertTensorBytes/Loads` fields are invalid and excluded from the comparison.
The primary storage, upload and disk counters are independent of that observer.
The harness now derives names from the canonical range plan and asserts binding.
A subsequent physical prefill verifies 96 tensor loads and 72,477,573,120 expert
payload bytes (`routed-cost-bank-counter-check-*`). Final-source lifecycle and
long-prefill replays are retained separately; their logits remain bit-exact. The
final source also hardens device-loss cleanup without changing successful math.

Two setup failures are retained: a missing isolated-snapshot helper, and the
initial planner's rejection of partial Q4K rows. The latter led to a per-row
padding correction and regression test, before parity succeeded. No failure was
converted into acceptance or a tolerance change.

Focused acceptance: cache/profile and old paging/preparation/alias tests; source
typing and unused/style gates; architecture/dependency views; browser import and
config single-source checks; package closure inventory. Four new runtime modules
with declarations and one profile are inventoried; no budget ceiling was raised.
No serving/catalog promotion, INT8 work, F32 KV experiment or correctness-policy
change is included.
