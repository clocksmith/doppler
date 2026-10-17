# Shipped prepared execution: AMD follow-up

Source: `64e27ed4`, isolated archive on the Radeon 8060S Vulkan host. No change to
its main checkout or installed contributor. This comparison was already running
when the work sequence moved to Reploid first; no additional candidate was made.

Qwen3.5 0.8B, AMD Radeon 8060S (RADV STRIX_HALO), Chrome browser WebGPU.
Doppler Q4_K/f32 compute/f16 KV versus Transformers.js q4f16 ONNX. These are
optimized product formats, not identical quantized weight bytes. The receipt
contains pinned artifacts, browser identity, configuration and resource samples.
Raw 64-token prompt, 128 generated tokens, greedy, 2 warmups, 3 timed runs,
Doppler first. Both engines reported the same physical AMD adapter.

| Metric | Doppler | Transformers.js |
| --- | ---: | ---: |
| Local loading ms | 3459 | 4386.75 |
| Warm first token ms | 572.1 | 275.92 |
| Decode tokens/s | 20.07 | 21.91 |
| Complete answer ms | 6899.6 | 6170 |
| Token-delivery gap p95 ms | 730.5 | 57.01 |

**This run loses warm TTFT, decode, completion and delivery smoothness.** All 128
output tokens match on this workload; this is not general correctness evidence.
The harness reports claim-grade and local-comparable, with releaseClaimable false.
OS caches were not purged; loading is not a cold download measurement. The large
difference from earlier isolated phase runs is unresolved. Host contention or
configuration differences must be established from evidence, not assumed away.

Reproduce using the argument array in [vulkan-08-command.json](vulkan-08-command.json)
from source `64e27ed4`, adapting only local model/browser/output paths. The source
snapshot must include `models/catalog.json` as well as runtime and harness files;
an initial missing-catalog setup failed before any comparison and was repaired.
No models were downloaded. Keep each engine's reset semantics and record both
engine orders in the next matched campaign.

[Complete comparator receipt](vulkan-08/compare_latest.json) and
[command output](vulkan-08/vulkan-08-compare.log) retain the result. Previous Metal
results and scoped physical qualification remain in
[the preparation report](../qwen-prepared-execution-20261017/README.md).

Next, after Reploid's physical search journey: trace browser prefill and delivery,
make at most one correction, then compare both 0.8B and 2B on Metal and Vulkan
while preserving throughput and parameter reuse. This receipt does not establish
an improvement over the former Doppler source or qualify the separate Vulkan
application/presentation proof.

Component: Doppler comparison evidence. Intent: preserved. Acceptance evidence:
local comparator receipt with exact workload token agreement and disclosed losses.
Boundary effects: none; no runtime or profile changes.
