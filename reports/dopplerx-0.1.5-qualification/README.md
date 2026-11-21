# DopplerX 0.1.5 Physical Qualification Bundle

This bundle retains physical hardware evidence for the `dopplerx@0.1.5` runtime across Apple M3 Metal and AMD Radeon 8060S Vulkan.

## 1. Cross-Vendor Tarball F16 Execution, Fail-Closed Precision, and Logical-Extent Gradient Clipping/Overflow

- [Apple M3 Metal receipt](mars-metal-f16.json)
- [AMD Radeon 8060S Vulkan receipt](jupiter-amd-f16.json)
- Tool: [verify-packed-f16-precision.js](../../tools/verify-packed-f16-precision.js)

Both receipts execute against the unpacked PR `#15` tarball (`SHA-256: ecb2a13e0823ea46fec3cccbf4aaae962ce412cd640eab4b56b69ac8499fde9a`, `2,059` files):
- **F16 compute & fail-closed precision:** Requested `shader-f16` compute dispatches return exact IEEE-754 half-precision words `[0x5c00, 0x6000]` (`[256, 512]`) across all four small-model manifests (`0.8B` and `2B`, legacy and reissued), while capability-disabled checks reject before dispatch instead of silently widening.
- **Physical GPU gradient clipping & overflow on dirty pooled buffers:** Exercises `clipGradients` and `detectOverflow` on physical Metal and Vulkan pooled GPU buffers containing trailing `NaN`, `+Inf`, and `200` values beyond logical tensor shape `[2]` for both `f32` (`clippedGpuValues: [0.6, 0.8]`, `unclippedNorm: 5`, `clippedNorm: 1`, `overflowWithDirtyPadding: false`, `overflowWhenLogicalElementNaN: true`) and `f16` (`clippedGpuF16Bits: [0x38cd, 0x3a66]`, `overflowWithDirtyPadding: false`, `overflowWhenLogicalElementNaN: true`).

## 2. Standalone Loopback HTTP Hybrid Prefix Reuse, Isolation, and Eviction (`doppler-serve`)

- [Qwen 3.5 0.8B throughput receipt](mars-metal-prefix-reuse-0-8b.json): `3.25x` follow-up prefill speedup (`1,573.14 ms` -> `483.96 ms`, saving `1,089.18 ms` prefill / `1,098.28 ms` wall time across two follow-up turns), `146,701,640` bytes retained (`96.00 MiB` contiguous KV + `43.91 MiB` Gated DeltaNet recurrent/conv checkpoints), `187` and `226` tokens reused, exact token parity across all turns.
- [Qwen 3.5 2B throughput receipt](mars-metal-prefix-reuse-2b.json): `4.06x` follow-up prefill speedup (`3,770.65 ms` -> `927.97 ms`, saving `2,842.68 ms` prefill / `2,875.40 ms` wall time across two follow-up turns), `146,701,640` bytes retained, `187` and `226` tokens reused, exact token parity across all turns.
- [Integrated `doppler-serve` multi-conversation isolation & LRU eviction receipt](mars-metal-serve-isolation-eviction.json): Interleaves three distinct conversations (`alpha`, `bravo`, `charlie`) through the live HTTP server under a bounded `320 MiB` (`335,544,320` byte) pool (`2` entries max at `293,400,560` allocated bytes). Verifies `3` isolated prefix hits (`0` cross-conversation leakage), `2` LRU evictions when `charlie` and returning `alpha` exceed pool capacity, and `100%` exact token ID parity against an uncached `--no-prefix-cache` baseline before and after eviction.
- Tools: [bench-serve-prefix-reuse.js](../../tools/bench-serve-prefix-reuse.js), [verify-serve-isolation-eviction.js](../../tools/verify-serve-isolation-eviction.js)

## 3. Routed-Expert-First Idle Piece Verification and Teardown Safety

- Implementation: [verified-piece-state-cache.js](../../src/experimental/distribution/verified-piece-state-cache.js), [verified-piece-gather-worker.js](../../src/tooling/node-shard-io/verified-piece-gather-worker.js), [mapped-expert-staging.js](../../src/inference/pipelines/text/mapped-expert-staging.js)
- Tests: [expert-priority-background-verify.test.js](../../tests/storage/expert-priority-background-verify.test.js), [mapped-expert-staging.test.js](../../tests/loader/mapped-expert-staging.test.js)

When `backgroundVerify` is enabled on verified piece storage, idle verification prioritizes routed MoE expert byte spans within the active layer partition ahead of non-expert prefix pieces, directly targeting the cold post-open routed-expert SHA-256 wait. Mapped expert staging also aborts in-flight `verifyQueued` reads when `destroy()` runs.

## 4. Composable JavaScript Capabilities and Choice Calibration (Node + Browser)

- [Node + Chromium WebGPU receipt](mars-metal-js-capabilities.json)
- Calibration implementation: [choice.js](../../src/client/doppler-api/operations/choice.js)
- Tool: [verify-js-capabilities.js](../../tools/verify-js-capabilities.js)

Verified sequentially on Apple M3 Metal across both Node WebGPU and Chromium WebGPU:
- **Generation** (`qwen-3-5-0-8b-q4k-ef16-af32-vf16`): cold load, warm decode, mid-stream `AbortController` cancellation settlement, deterministic post-cancel reuse on the same handle, and `0` active GPU buffers after `unload()`.
- **Embeddings and Dual-Resident Search Reranking** (`qwen-3-embedding-0-6b-q4k-ehf16-af32` + `qwen-3-reranker-0-6b-f16-af32`): `6/6` exact top-1 document matches on the reference corpus while both models remain simultaneously resident on GPU (`2.21 GB` combined GPU allocation), pre-flight cancellation rejection, and independent `unload()` cleanup.
- **Typed Choice Scoring and Empirical Temperature Calibration** (`qwen-3-reranker-0-6b-f16-af32`): `12/12` relevance selections matching reference answers on physical GPU logits; empirical temperature scaling (`fitChoiceTemperatureCalibration` / `applyChoiceTemperatureCalibration`) fitted on `cases[0..5]` reduced held-out `cases[6..11]` log-loss (`0.004835` -> `0`) and ECE (`0.004809` -> `0`) while preserving raw logits and `1.0` accuracy.
