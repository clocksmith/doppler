# Warm comparison harness audit — 2026-10-17

Component: `doppler.runtime-source.inference`, `doppler.benchmarks.vendors`.
Intent: preserved. Boundary effects: benchmark timing contract and comparator
runner; inference kernels and model artifacts are unchanged.

## Reproducible observation

[Command](command-metal-08.json), [source hashes](source.json),
[receipt](metal-08-tjs-first/compare_latest.json), [raw log](metal-08.log).

Qwen3.5 0.8B on Apple M3 Metal, local RDRR Q4_K with f16 embeddings/KV
and f32 compute versus pinned Transformers.js 4.2.0 ONNX q4f16. These are
optimized product formats, not identical quantized tensors. Raw synthetic
64-token prompt, greedy 128-token output, one warmup, three timed requests,
HTTP model loading, Transformers.js first, reused engine and fresh sequence
state per request. Source base 43787cf7 plus the captured harness changes.

| Metric (median) | Doppler | Transformers.js |
| --- | ---: | ---: |
| Warm first-token latency | 272.70 ms | 137.73 ms |
| Decode (127 tokens) | 54.47 tok/s | 49.64 tok/s |
| Model-load observation | 1494.20 ms | 2320.00 ms |
| Total generation request | 2609.60 ms | 2694.93 ms |
| Inter-token p95 arrival gap | 284.50 ms | 24.52 ms |

The output-parity gate passes for this workload. It is not general model
correctness. Three samples establish a physical harness check, not a narrow-win
promotion or a cross-device claim. The load observation is not a purged OS-cache
or fresh-install cold-start measurement. The much faster timings than earlier
sessions cannot be attributed to this observation-only change: both engines
improved, and the host was not exclusively reserved.

## Fixes and open work

- Doppler no longer silently filters timed stalls with IQR. The older three-run
  receipts were unaffected by that filter because it requires four samples.
- Comparable timing includes generation entry, tokenization, token callbacks,
  and completion. Internal phases remain independently inspectable.
- Raw request samples survive; empty/mismatched/non-finite timing cannot pass.
- Warm engine order is explicit and can alternate between paired sessions.
- Warm TTFT and token delivery tails remain losses. No performance optimization
  is promoted by this audit. AMD rerun, adjacent counterbalanced campaigns,
  fresh-browser startup and longer-context prefill remain open.
- Community local and pooled LoRA training is a separate authorized capability
  task; this timing receipt supplies no training evidence.

## Acceptance evidence

Focused CPU tests: browser-harness-request-timing, browser-harness-batching-metrics,
browser-harness-generation-diagnostics, browser-harness-reference-transcript,
and compare-engines-cli-contract. Config single-source check passes.
The new test injects first-token stalls, empty decoded chunks, completion cost,
and invalid comparison samples. It does not simulate GPU correctness.

Previous skipped CI and qualification failures remain open; these checks do not
close them.

## Prefill attribution after the harness repair

[Instrumented command](prefill-profile-command.json),
[config](prefill-profile-config.json), [raw receipt](prefill-profile.log), and
[summary](prefill-attribution.json) retain one warm timestamp-instrumented
64-input/128-output request. Observed request TTFT was 250.7 ms, internal prefill
247.4 ms, GPU timestamp sum 175.4 ms, CPU recording 47.9 ms, and submit/wait
153.5 ms. GPU work: projections 135.2 ms, recurrent attention 30.9 ms, full
attention 4.37 ms, convolution 3.37 ms. There were 337 recorded operations.

These clocks are not a disjoint additive partition: submit/wait includes GPU
execution, and CPU recording can overlap queued work. They identify projections
and command recording as the main targets without claiming that the arithmetic
residual is pure overhead. The instrumented request is excluded from primary
performance statistics. No kernel selection was changed or promoted.

Next experiment: attribute recording cost within projection dispatch and test
one projection or recording change against the unchanged graph using adjacent
paired samples and numerical gates. The earlier narrow-tile candidate was
neutral in the browser and remains unadopted. Warm TTFT, burst-delivery tails,
AMD timing reruns, 2B current-harness coverage, cold-start separation and transfer
latency remain open. Current community training qualification is retained
[separately](../community-lora-20261017/README.md).
