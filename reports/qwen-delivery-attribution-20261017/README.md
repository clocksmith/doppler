# Qwen token-delivery attribution — 2026-10-17

The 16-token bursts originate before the consumer: Doppler waits for a complete
GPU token group, then yields its tokens together. Reducing readback grouping
improves cadence but loses throughput on both GPUs. **The candidate is rejected;
no runtime/profile change is retained.** This follows the [frozen package and
matched comparisons](../qwen-sampling-correction-20261017/README.md).

Runtime archive SHA-256:
`75f3a81f032e3a0f89102fb9a50e0b202657f69eab71569585d7ffe4ce232bb2`.
Physical scope: Qwen3.5 0.8B, `qwen-3-5-0-8b-q4k-ehaf16`, Q4_K/F32 compute/F16 KV,
M3 Metal and Radeon 8060S Vulkan, the same browsers and pinned model manifests as
the matched comparison. Raw 64-token input, 128 greedy output tokens; two warmups,
three untraced requests, then three instrumented requests per invocation. Every
request checks all generated token IDs against its retained reference. Every
served runtime file is verified against the immutable archive inventory.

[Summary](summary.json), [reducer](summarize.py), and [probe](browser-trace.mjs)
retain exact source hashes, receipts, per-request timings and coverage. These are
controlled diagnostic measurements with a DOM/stream fixture, not replacement
Transformers.js comparison numbers or Reploid rendering evidence.

## One-variable experiment

Keep submission batch size four. Change only
`inference.session.decodeLoop.readbackInterval` from four to one, yielding groups
of 16 versus four tokens. The existing runtime remains byte-identical. Execute
baseline → candidate → restored baseline separately on each host. Table entries
are medians of three untraced requests, with no outlier removal. Delivery p95 uses
linear interpolation across actual inter-yield gaps, then the median across runs.

| GPU | Configuration | Decode tokens/s | Delivery p95 ms | Complete ms |
| --- | --- | ---: | ---: | ---: |
| M3 | baseline before | 58.84 | 265.21 | 2328.8 |
| M3 | four-token candidate | 50.32 | 79.90 | 2689.4 |
| M3 | baseline restored | 58.65 | 265.65 | 2337.3 |
| AMD | baseline before | 113.09 | 135.43 | 1243.0 |
| AMD | four-token candidate | 76.30 | 53.51 | 1778.6 |
| AMD | baseline restored | 114.50 | 134.31 | 1231.1 |

Throughput regresses roughly 14% on M3 and 33% on AMD against restored controls.
First-token latency barely changes: the first token is yielded before grouped
decode. This experiment does not solve the separate prefill/TTFT losses.

## Attribution

The corrected probe wraps the live device queue, buffer mapping, the grouped
decode method, and the JavaScript sampling method. It records stream enqueue,
consumer dequeue, DOM append and animation-frame callbacks without awaiting a
frame per token. Missing submission/map/group observations fail the probe.

Each traced request contains one initial token followed by seven groups of 16
and one group of 15. There are 32 decode submissions, because submission still
happens every four tokens. Token copies/readback occur after the entire group;
the generator awaits that result before yielding any member. See
[submission and readback](../../src/inference/pipelines/text/generator-steps.js)
and [await/yield ownership](../../src/inference/pipelines/text/generator/decode-runtime.js).

| Median traced component | M3 ms | AMD ms |
| --- | ---: | ---: |
| Decode wall | 2172.6 | 1105.2 |
| Command recording | 455.5 | 572.9 |
| Readback map wait | 1705.2 | 505.3 |
| Readback cleanup | 8.7 | 18.2 |
| Readback CPU copy | 0.1 | 0.2 |
| Synchronous queue-submit calls, whole request | 0.3 | 1.1 |
| JavaScript sampling, initial token only | 1.1 | 1.5 |
| Maximum stream yield-to-dequeue delay | 0.1 | 0.2 |
| Animation-frame opportunity p95 after DOM update | 61.1 | 81.1 |

Map wait includes queued GPU execution and completion notification; it is **not**
pure transfer cost. Submission/completion waits overlap and must not be summed.
No GPU timestamp decomposition was captured. The stream fixture adds no token
batching. DOM appends are immediate, but command recording can delay animation
frames; a callback is an opportunity to present, not proof that pixels appeared.
No JavaScript sampling occurs for grouped decode in this greedy GPU lane.

The four-token candidate increases map wait to 2132.7 ms on M3 and 1043.2 ms on
AMD. It creates more serial host/GPU handoffs. A future delivery change would need
to expose completed prefixes while retaining queued work and resource ownership;
this is a hypothesis, not an implemented or qualified optimization.

## Retention and next boundary

The null-seed invocation and wrong stats-accessor attempts are retained as probe
setup failures. Earlier successful traces missed queue submissions because the
runtime already owned an instance wrapper; they remain retained with their exact
probe source. `*-queue-observed.json` supplies complete instrumentation coverage.
No zero-observation counter is interpreted as zero submission cost.

Raw receipts, logs and all probe versions are in `raw-evidence.tar.gz`; extract
in this directory. `SHA256SUMS` binds that bundle and the readable report. To
reproduce, run `node browser-trace.mjs SNAPSHOT MODEL_DIR BASELINE_COMPARE_JSON
FRESH_OUTPUT_JSON [READBACK_INTERVAL]`, using the archived runtime and provisioned
dependencies. Omit the final argument for baseline; use `1` for the rejected
candidate. `summarize.py` verifies served bytes and exact probe hashes.

This is the stopping point after package qualification, matched measurements,
and delivery attribution. Per-tensor/alias reachability remains unimplemented;
next compile immutable requirements including reachable fallbacks, then qualify
27B Q4 memory and 2B control before reusable bindings and scratch arenas. The
[single optimization inventory](../../docs/performance/potential-optimizations.md)
remains the owner of those proposals. No WAN, LoRA, prefix-cache or model expansion.

Component: `doppler.repository-tooling`, `doppler.docs`. Intent: preserved.
Acceptance: exact-token diagnostic runs on both GPUs, frozen-source verification,
baseline/candidate/restored-control receipts. Boundary effects: none. This does
not close earlier CI failures, establish full release acceptance, or promote an
artifact.
