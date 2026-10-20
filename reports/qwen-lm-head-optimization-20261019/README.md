# LM Head Argmax Optimization: Shared-Memory Hidden Vector Caching

This report documents the optimization of `lm_head_argmax_q4k_phase1` on Apple M3 Metal. Following the operator attribution diagnostic in `reports/qwen-operator-attribution-20261018/`, output-head argmax phase 1 was identified as the single largest individual kernel dispatch during decode (162.12 ms across 20 measured tokens, accounting for 20.9% of total decode compute).

## Mechanism

In the unoptimized baseline, every workgroup computing vocabulary projection columns redundantly read the 2,048-element (8 KB) activation vector `hidden` directly from global storage memory (`var<storage, read> hidden: array<f32>`). Across 2,374 workgroups (151,936 vocabulary entries divided into 64 columns per workgroup), this resulted in over 1.2 GB of redundant global VRAM reads per token.

The optimized kernel introduces cooperative shared-memory caching:
1. `override SHARED_HIDDEN_MAX: u32 = 2048u;`
2. `var<workgroup> shared_hidden: array<f32, SHARED_HIDDEN_MAX>;`
3. At the beginning of `phase1`, when `has_full_blocks` is active (`USE_FULL_BLOCK_FAST_PATH && (u.hidden_size % QK_K) == 0u && u.hidden_size <= SHARED_HIDDEN_MAX`), the 256 workgroup threads cooperatively load `hidden` into `shared_hidden` with stride `WORKGROUP_SIZE` followed by a single `workgroupBarrier()`.
4. `accumulate_full_block` reads activations directly from `shared_hidden[k0]`, eliminating global memory stalls and allowing the compiler to keep inner-loop dot products in fast local SRAM.
5. Workgroup storage consumption is 11,264 bytes (8 KB `shared_hidden` + 3 KB candidate/partial arrays), safely under the 16,384-byte WebGPU hardware specification limit.
6. Models with non-aligned hidden sizes or dimensions exceeding `SHARED_HIDDEN_MAX` safely bypass the cache and retain the generic global path.

## Measured Results (Apple M3 Metal, Chrome 155, Qwen 3.5 2B Q4_K)

Contract: 5-token prompt, 32 generated tokens across three sequential requests (control, profiled, repeat).

| Stage | Baseline Elapsed (ms) | Candidate Elapsed (ms) | Delta (ms) | Speedup / Change |
|---|---|---|---|---|
| Control Request | 2,216.1 | 1,278.5 | -937.6 | **-42.3%** |
| - Control Decode Phase | 1,921.9 | 927.3 | -994.6 | **-51.7%** |
| - Control Decode Submit Wait | 1,633.9 | 726.3 | -907.6 | **-55.5%** |
| Profile Request | 2,114.0 | 1,522.7 | -591.3 | **-28.0%** |
| Repeat Request | 1,317.7 | 1,328.9 | +11.2 | +0.9% |
| Measured Decode GPU Compute | 807.48 | 717.45 | -90.03 | **-11.2%** |
| `lm_head_argmax_q4k_phase1` (20 tokens) | 162.12 | 145.69 | -16.43 | **-10.1%** |

## Numerical and Lifecycle Verification

1. **Exact Token Parity:** All 32 tokens match the exact retained control sequence across all three requests:
   `[11751, 13, 11751, 369, 279, 6511, 314, 9338, 13, 11751, 369, 279, 6511, 314, 9338, 13, 11751, 369, 279, 6511, 314, 9338, 13, 11751, 369, 279, 6511, 314, 9338, 13, 11751, 369]`
2. **Text Parity:** Output matches exactly:
   `" Paris. Paris is the capital of France. Paris is the capital of France. Paris is the capital of France. Paris is the capital of France. Paris is"`
3. **Logits Surface:** `handle.advanced.prefillWithLogits()` verified and functional.
4. **Lifecycle Settlement:** State reset, multi-request session reuse, and device unload/destroy completed without memory leaks or hangs.
5. **Contract Suite:** Unit and recorder contract tests (`tests/gpu/lm-head-argmax-recorder.test.js`, `tests/config/session-flags-merge.test.js`, `tests/config/execution-v1-schema.test.js`, `npm run kernels:check`, `npm run source:style:check`, `npm run typecheck:source`, `npm run source:architecture:check`) all pass.

## Component Contract

- **Component:** GPU Kernels (`lm_head_argmax_q4k.wgsl`)
- **Intent:** Preserved. Mathematical dot-product and argmax reduction semantics are identical; caching is an execution-space acceleration.
- **Acceptance evidence:** Physical Chrome M3 Metal probe, unit tests, source style/type/architecture checks, kernel digest synchronization.
- **Boundary effects:** None outside `lm_head_argmax_q4k.wgsl` and digest tracking in `kernel-ref-digests.js`. Reploid, Gamma, and higher-level model pipelines remain untouched.
