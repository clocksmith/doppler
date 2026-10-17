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
