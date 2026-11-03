# Shared-model performance diagnostics, 2026-11-03

This is diagnostic evidence, not model qualification or a performance claim. No runtime change was made. The source and manifest identities, task configuration, per-attempt outputs, logs and memory samples are retained in the linked evidence directories. The previously published 35B and 27B coverage gaps and losses remain open.

## 27B current-source screen on Jupiter

On AMD Ryzen AI MAX+ 395 / Chrome WebGPU, the short prompt (28 input tokens, 128 output cap, greedy) was run with one warmup and two measured requests per engine. Doppler ran commit `26c3fccfb52af6c4e239e6a07a675224692af146`, with the retained Q5_K blockwise candidate configuration; Pooled ran its native MTP depth 3, lookup off, from clean source commit `6b4a24f73cce7ef9a9e3ab35b9a0d4c09b51dfff`. Both engines remained resident and the measured order alternated. Results are complete-request milliseconds:

| Engine | Warmup | Measured 1 | Measured 2 |
| --- | ---: | ---: | ---: |
| Doppler | 12,495.2 | 11,126.7 | 11,287.6 |
| Pooled | 6,971.7 | 7,074.8 | 7,063.9 |

All six requests returned the same 128 token IDs and recorded no GPU errors. Doppler's measured prefill was 998.8/1,104.1 ms and decode was 10,116.7/10,159.2 ms. Pooled recorded 594.0/602.4 ms prefill; each measured request accepted 81 of 139 drafts. Readiness took 52,048.6 ms for Doppler and 104,743 ms for Pooled in this serial loading order. Pooled's memory field was absent; do not infer equal memory or resource costs. The native engines use different materialization and cache arithmetic, and agreement on one fixed prompt is not an independent numerical gate or held-out task-quality qualification. There is no long-prompt result in this run.

The RDRR manifest SHA-256 is `69b8c4e9aa0fb148581595d3da8b0cb304a3eb942e1da0a2c7267e73c86662de`. The Pooled Qwen engine file SHA-256 is `2e126c9c401e02f1ee1ab13a522228ef02e663f5973bea17bc7fa2d62faaa064`. The [Jupiter evidence](jupiter/) retains the exact config, receipt, outputs, source snapshots of the harness and readiness metadata; large files are gzip-compressed. Pooled readiness is summarized and the full original is identified by hash to avoid retaining model metadata. The receipt's inherited scope mentions one measured pair, but the config and six attempts show two measured requests per engine. This is an exploratory concurrent-residency screen, not a replacement for an isolated matched-numerics lane.

## Mars 27B timeout investigation

One short 128-token current-source warmup previously exceeded 240 seconds; no measured request or receipt was obtained and the process was stopped after cleanup stalled. To isolate it, four-token Chrome diagnostics used the same prompt, one engine resident at a time, a 120-second request limit, progress logging, a receipt written before teardown, an external process-group watchdog and the cooperative GPU lock. The first and third controls used the earlier `runtime-04` source snapshot, whose recorded base commit was `727d36aa0f0e8705d4f3121ae727e463de4a9d6f`. The other two used frozen `26c3fccf`, with whole-token recording enabled and disabled respectively. These are cold, instrumented requests, not steady-state comparisons.

| Run | Result | First token | Complete | Notes |
| --- | --- | ---: | ---: | --- |
| Older snapshot, first | 4 tokens | 46.29 s | 58.81 s | 44.03 s prefill submission wait; 12.41 s decode readback wait |
| Current source | Timeout | 47.62 s | >120 s | No second token delivered |
| Older snapshot, repeat | 4 tokens | 19.43 s | 61.28 s | 18.49 s prefill submission wait; 41.72 s decode readback wait |
| Current source, whole-token recording off | Timeout | 48.14 s | >120 s | `context.close()` stalled; watchdog stopped the process group |
| Pooled, native MTP | 4 tokens | 128.79 s | 130.10 s | 180-second request limit; different arithmetic and independent engine run |

The completed runs returned the same four token IDs. The 24 GiB machine showed substantial swap use and swap I/O in the retained samples, alongside highly variable GPU waits. Swap is an observation, not an established cause. The two current-source timeouts are a regression signal, not proof of a source regression: samples are few and load, memory pressure and order vary. Disabling whole-token recording did not remove the timeout. The diagnostic wrapper preserved the timeout receipt before teardown; the browser-context close still hung in one run, so the external watchdog terminated its own process group. Mars's GPU lock was confirmed free afterward. [Mars evidence](mars/) contains the configs, logs, receipts, memory samples, harness snapshot and launcher scripts. These diagnostic changes were confined to a temporary harness; they are not a production cleanup fix.

## Remaining work

Native MTP for 35B MoE is not implemented: auxiliary tensors have been prepared, but draft execution, target MoE batch verification, exact greedy acceptance, and recurrent/KV recovery including rejection, cancellation and continuation must be implemented and physically qualified. The shared Qwen path should be composable across the 27B dense and 35B MoE models, with declared model-specific operations and no implicit fallback. Stochastic speculation needs its own probability-correctness gate.

Reproduce and isolate the Mars current-source stall under controlled memory conditions; retain failed attempts and teardown behavior. Compare the proposed fast path against unchanged baselines with identical workload boundaries and declared numerical formats, then qualify both models with independent numerical references, held-out task checks, cancellation, cold/warm readiness and memory. The 35B/27B long-prompt and host/surface coverage remains as recorded in the [scoreboard](../README.md). No broad speed lead is established.

Component: `doppler`. Intent: preserved. Boundary effects: evidence only. Acceptance evidence: retained receipts and logs; no new runtime behavior or model qualification.
