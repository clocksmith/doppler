# DopplerX 0.1.5 Physical Qualification Bundle

This bundle retains physical hardware evidence for the `dopplerx@0.1.5` runtime across Apple M3 Metal and AMD Radeon 8060S Vulkan.

## 1. Cross-Vendor F16 Execution and Fail-Closed Precision

- [Apple M3 Metal receipt](mars-metal-f16.json)
- [AMD Radeon 8060S Vulkan receipt](jupiter-amd-f16.json)
- Tool: [verify-packed-f16-precision.js](../../tools/verify-packed-f16-precision.js)

Both vendors execute requested `shader-f16` compute dispatches returning IEEE-754 half-precision words `[0x5c00, 0x6000]` (`[256, 512]`) across all four small-model manifests (`0.8B` and `2B`, legacy and reissued), and reject unsupported precision configurations fail-closed instead of silently widening or narrowing.

## 2. Standalone Loopback HTTP Hybrid Prefix Reuse (`doppler-serve`)

- [Qwen 3.5 0.8B receipt](mars-metal-prefix-reuse-0-8b.json): `3.25x` follow-up prefill speedup (`1,573.14 ms` -> `483.96 ms`, saving `1,089.18 ms` prefill / `1,098.28 ms` wall time across two follow-up turns), `146,701,640` bytes retained (`96.00 MiB` contiguous KV + `43.91 MiB` Gated DeltaNet recurrent/conv checkpoints), `187` and `226` tokens reused, exact token parity across all turns.
- [Qwen 3.5 2B receipt](mars-metal-prefix-reuse-2b.json): `4.06x` follow-up prefill speedup (`3,770.65 ms` -> `927.97 ms`, saving `2,842.68 ms` prefill / `2,875.40 ms` wall time across two follow-up turns), `146,701,640` bytes retained, `187` and `226` tokens reused, exact token parity across all turns.
- Tool: [bench-serve-prefix-reuse.js](../../tools/bench-serve-prefix-reuse.js)

`estimateGenerationContextMinimumBytes` accounts for lazily created Gated DeltaNet recurrent and convolution checkpoints alongside contiguous KV allocations before turn 1 so hybrid models borrow the full `256 MiB` entry budget rather than overflowing a `128 MiB` half-pool reservation.

## 3. Routed-Expert-First Idle Piece Verification and Teardown Safety

- Implementation: [verified-piece-state-cache.js](../../src/storage/verified-piece-state-cache.js), [verified-piece-gather-worker.js](../../src/storage/verified-piece-gather-worker.js), [mapped-expert-staging.js](../../src/loader/experts/mapped-expert-staging.js)
- Tests: [expert-priority-background-verify.test.js](../../tests/storage/expert-priority-background-verify.test.js), [mapped-expert-staging.test.js](../../tests/loader/mapped-expert-staging.test.js)

When `backgroundVerify` is enabled on verified piece storage, idle verification prioritizes routed MoE expert byte spans within the active layer partition ahead of non-expert prefix pieces, directly targeting the cold post-open routed-expert SHA-256 wait. Mapped expert staging also aborts in-flight `verifyQueued` reads when `destroy()` runs.

## 4. Composable JavaScript Capabilities and Choice Calibration (Node + Browser)

- [Node + Chromium WebGPU receipt](mars-metal-js-capabilities.json)
- Calibration implementation: [choice-calibration.js](../../src/inference/choice-calibration.js)
- Tool: [verify-js-capabilities.js](../../tools/verify-js-capabilities.js)

Verified sequentially on Apple M3 Metal across both Node WebGPU and Chromium WebGPU:
- **Generation** (`qwen-3-5-0-8b-q4k-ef16-af32-vf16`): cold load, warm decode, mid-stream `AbortController` cancellation settlement, deterministic post-cancel reuse on the same handle, and `0` active GPU buffers after `unload()`.
- **Embeddings and Dual-Resident Search Reranking** (`qwen-3-embedding-0-6b-q4k-ehf16-af32` + `qwen-3-reranker-0-6b-f16-af32`): `6/6` exact top-1 document matches on the reference corpus while both models remain simultaneously resident on GPU (`2.21 GB` combined GPU allocation), pre-flight cancellation rejection, and independent `unload()` cleanup.
- **Typed Choice Scoring and Empirical Temperature Calibration** (`qwen-3-reranker-0-6b-f16-af32`): `12/12` relevance selections matching reference answers on physical GPU logits; empirical temperature scaling (`fitChoiceTemperatureCalibration` / `applyChoiceTemperatureCalibration`) fitted on `cases[0..5]` reduced held-out `cases[6..11]` log-loss (`0.004835` -> `0`) and ECE (`0.004809` -> `0`) while preserving raw logits and `1.0` accuracy.
