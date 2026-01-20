# DOPPLER Postmortems

Quick reference for debugging history and lessons learned.

---

## Summary Table

| Post-Mortem | Date | Status | Root Cause |
|-------------|------|--------|------------|
| [Gemma 3 Post-Feedforward Norm](#2026-01-19-gemma3-postfeedforwardnorm) | 2026-01-19 | RESOLVED | Missing `postFeedforwardNorm: true` in preset - weights existed but norm was skipped |
| [Gemma 2 F16 End-to-End Pipeline](#2026-01-05-gemma2-f16-end-to-end) | 2026-01-05 | RESOLVED | F16 activations not fully plumbed through attention/logits/sampling + kernel path selection ignored activation dtype |
| [Gemma 2 Activation Dtype Mismatch](#2026-01-04-gemma2-activation-dtype-mismatch) | 2026-01-04 | RESOLVED | F16 matmul override on F32 activations + phase-agnostic attention lookup |
| [Performance Gaps (F16)](#2026-01-03-performance-gaps) | 2026-01-03 | **OPEN** | Matmul/attention throughput still behind WebLLM; F16 activations now end-to-end |
| [Batched Decode Repetition](#2026-01-03-batched-decode-repetition) | 2026-01-03 | **OPEN** | Uniform cache eviction destroys pending buffers |
| [Ping-Pong Buffer Release](#2025-12-31-pingpong-buffer-release) | 2025-12-31 | RESOLVED | Index-based lookup missed buffer B after odd layers |
| [Buffer Pool Trace False Positives](#2025-12-30-buffer-pool-trace-false-positives) | 2025-12-30 | RESOLVED | Trace reading garbage from buffer pool padding |
| [Gemma 3 q_norm/k_norm Offset](#2025-12-25-gemma3-qknorm-offset) | 2025-12-25 | RESOLVED | Missing +1 offset for q_norm/k_norm weights |
| [MoE Explicit Layout](#2025-12-22-moe-explicit-layout) | 2025-12-22 | RESOLVED | WebGPU 'auto' layout binding mismatch |
| [Gemma 3 Magnitude Gap](#2025-12-21-gemma3-magnitude-gap) | 2025-12-21 | SUPERSEDED | Superseded by q_norm/k_norm fix |
| [BF16 2D Dispatch](#2025-12-20-bf16-2d-dispatch) | 2025-12-20 | RESOLVED | 2D dispatch without linearization |
| [Pipeline Verification](#2025-12-19-pipeline-verification) | 2025-12-19 | RESOLVED | Identified FFN explosion |
| [Gemma 3 Q4K Garbage Output](#2025-12-18-gemma3-q4k-garbage-output) | 2025-12-18 | RESOLVED | Q4K layout mismatch + q_norm offset |
| [Hidden State Under-Accumulation](#2025-12-18-hidden-state-underaccumulation) | 2025-12-18 | SUPERSEDED | Merged into Garbage Output PM |
| [Positive Bias Hidden States](#2025-12-17-positive-bias-hidden-states) | 2025-12-17 | DISPROVED | Sampling artifact, not real issue |
| [Softmax Uniform Buffer](#2025-12-17-softmax-uniform-buffer) | 2025-12-17 | RESOLVED | Swapped innerSize/outerSize |
| [Gemma 3 Debug](#2025-12-16-gemma3-debug) | 2025-12-16 | RESOLVED | Q4K quantization format mismatch |

---

## Post-Mortem Details

### 2026-01-19-gemma3-postfeedforwardnorm

**Status:** RESOLVED | **File:** [2026-01-19-gemma3-postfeedforwardnorm.md](postmortems/2026-01-19-gemma3-postfeedforwardnorm.md)

Gemma 3 produced gibberish (token 138 repeated). Root cause: `postFeedforwardNorm: false` in preset but model has `post_feedforward_layernorm.weight` tensors. Sandwich norm (Peri-LN) requires normalizing FFN output. Fix: set `postFeedforwardNorm: true` and added tensor-config consistency validator.

---

### 2026-01-05-gemma2-f16-end-to-end

**Status:** RESOLVED | **File:** [2026-01-05-gemma2-f16-end-to-end.md](postmortems/2026-01-05-gemma2-f16-end-to-end.md)

F16 activations were enabled but attention, RoPE, logits, and sampling still assumed F32. Fixes added F16 kernels and dtype propagation, plus explicit kernel-path defaults from `kernelPaths` + `quantizationInfo.compute`.

---

### 2026-01-04-gemma2-activation-dtype-mismatch

**Status:** RESOLVED | **File:** [2026-01-04-gemma2-activation-dtype-mismatch.md](postmortems/2026-01-04-gemma2-activation-dtype-mismatch.md)

Gemma 2 debug runs produced incoherent output. Root cause: kernel path overrides forced F16 outputs while runtime activations remained F32, and attention variant selection ignored prefill/decode phase. Fixes made kernel selection phase-aware, enforced dtype matches, and defaulted activations to F16.

---

### 2026-01-03-performance-gaps

**Status:** OPEN | **File:** [2026-01-03-performance-gaps.md](postmortems/2026-01-03-performance-gaps.md)

DOPPLER decode remains slower than WebLLM on Gemma 2 2B. F16 activations are now end-to-end; remaining gap is dominated by matmul and attention throughput (see profiling in the postmortem).

---

### 2026-01-03-batched-decode-repetition

**Status:** OPEN (workaround applied) | **File:** [2026-01-03-batched-decode-repetition.md](postmortems/2026-01-03-batched-decode-repetition.md)

Batched decode (batchSize > 1) produces repetitive output after ~12 tokens while single-token decode works correctly. Root cause hypothesis: uniform buffer cache (256 max entries) evicts LRU entries and DESTROYS the GPU buffer while still referenced by pending command buffers. Workaround: set `runtime.inference.batching.batchSize = 1` via runtime preset/override. Fix requires deferred buffer destruction after command buffer completion.

---

### 2025-12-31-pingpong-buffer-release

**Status:** RESOLVED | **File:** [2025-12-31-pingpong-buffer-release.md](postmortems/2025-12-31-pingpong-buffer-release.md)

Decode regression after ping-pong buffer optimization. Two pre-allocated buffers A and B alternate as input/output across layers. After odd layers, buffer release check used `getHiddenBuffer()` which returns index-dependent value, causing B to be incorrectly released. Fix: capture both buffers upfront before the layer loop.

---

### 2025-12-30-buffer-pool-trace-false-positives

**Status:** RESOLVED | **File:** [2025-12-30-buffer-pool-trace-false-positives.md](postmortems/2025-12-30-buffer-pool-trace-false-positives.md)

Kernel trace showed alarming "explosions" during decode. False positives caused by trace reading garbage from buffer pool padding. Fix: use shape to bound iteration in debug readbacks.

---

### 2025-12-25-gemma3-qknorm-offset

**Status:** RESOLVED | **File:** [2025-12-25-gemma3-qknorm-offset.md](postmortems/2025-12-25-gemma3-qknorm-offset.md)

Gemma 3 produced garbage output. Root cause: q_norm and k_norm weights were missing the +1 offset that ALL Gemma 3 RMSNorm layers require. Fix: use `tryLoadNorm()` for q_norm/k_norm with `getNormWeightBuffer()` in attention.js.

---

### 2025-12-22-moe-explicit-layout

**Status:** RESOLVED | **File:** [2025-12-22-moe-explicit-layout.md](postmortems/2025-12-22-moe-explicit-layout.md)

MoE gather kernel compiled but didn't execute - `tokenCounts` was all zeros. WebGPU's `layout: 'auto'` creates layout with only used bindings, causing silent mismatch. Fix: create explicit bind group layout with all 6 bindings.

---

### 2025-12-21-gemma3-magnitude-gap

**Status:** SUPERSEDED | **File:** [2025-12-21-gemma3-magnitude-gap.md](postmortems/2025-12-21-gemma3-magnitude-gap.md)

Found GPU sync bug in `scaleGPUBuffer()`. The remaining gap was later resolved by the q_norm/k_norm +1 offset fix (see 2025-12-25-gemma3-qknorm-offset.md).

---

### 2025-12-20-bf16-2d-dispatch

**Status:** RESOLVED | **File:** [2025-12-20-bf16-2d-dispatch.md](postmortems/2025-12-20-bf16-2d-dispatch.md)

Large vocab models produced garbage output. BF16->F32 kernel used 2D dispatch but only used `global_id.x`, ignoring `global_id.y`. Fix: compute linear index from 2D dispatch using `workgroupsX` uniform.

---

### 2025-12-19-pipeline-verification

**Status:** RESOLVED | **File:** [2025-12-19-pipeline-verification.md](postmortems/2025-12-19-pipeline-verification.md)

Systematic verification identified FFN down projection explosion causing near-uniform logits. The explosion was later traced to Q4K quantization format mismatch.

---

### 2025-12-18-gemma3-q4k-garbage-output

**Status:** RESOLVED | **File:** [2025-12-18-gemma3-q4k-garbage-output.md](postmortems/2025-12-18-gemma3-q4k-garbage-output.md)

Q4K weights stored in packed 256-block stream incompatible with row-wise fused matmul addressing. Fix: loader fallback dequantizes packed Q4K to F16 for correctness.

---

### 2025-12-18-hidden-state-underaccumulation

**Status:** SUPERSEDED | **File:** [2025-12-18-hidden-state-underaccumulation.md](postmortems/2025-12-18-hidden-state-underaccumulation.md)

Superseded by 2025-12-18-gemma3-q4k-garbage-output which identified the actual root cause.

---

### 2025-12-17-positive-bias-hidden-states

**Status:** DISPROVED | **File:** [2025-12-17-positive-bias-hidden-states.md](postmortems/2025-12-17-positive-bias-hidden-states.md)

"Positive bias" was a sampling artifact - debug only read 5 values, not full 1152-dim vector. Investigation fixed real bugs (attention variant selection, workgroup dispatch, debug readback timing).

---

### 2025-12-17-softmax-uniform-buffer

**Status:** RESOLVED | **File:** [2025-12-17-softmax-uniform-buffer.md](postmortems/2025-12-17-softmax-uniform-buffer.md)

Softmax kernel failed with maxError=0.137. TypeScript wrote uniform fields in wrong order vs WGSL struct layout. Fix: corrected uniform buffer writes.

---

### 2025-12-16-gemma3-debug

**Status:** RESOLVED | **File:** [2025-12-16-gemma3-debug.md](postmortems/2025-12-16-gemma3-debug.md)

Model output `<unused16>` tokens. Root cause: quantizer used wrong format - `q * scale + min` instead of llama.cpp's `d * sc * q - dmin * min`. Fix: rewrote `quantizeQ4KBlock()` to match llama.cpp byte layout.

---

## Common Patterns

### Config-Tensor Mismatch (NEW)
- Preset/manifest sets feature flag to `false` but model has the weight tensors
- Validation only checked schema, not tensor presence
- Fix: tensor-config consistency validator now fails on mismatch
- Prevention: auto-detect normalization flags from tensor names in converter

### Buffer Lifecycle Bugs
- Ping-pong release: index-based lookup missed alternate buffer after swap
- Uniform cache eviction: destroying buffers still referenced by pending command buffers
- Defer buffer destruction until after GPU command completion

### Silent WebGPU Failures
- MoE explicit layout: no error, kernel just didn't run
- BF16 2D dispatch: no error, partial data processed

### Quantization Format Bugs
- Q4K packed vs row-wise layout
- Scale/min encoding mismatches

### Architecture-Specific Weight Processing
- Gemma 3: ALL norms use `(1 + weight)` including q_norm/k_norm
- Always verify against HuggingFace source for architecture quirks

### Uniform Buffer Mismatches
- TypeScript/WGSL struct field order must match exactly

### Debug Instrumentation Errors
- Sampling only first N values hides real distribution
- Buffer pool padding contains garbage - always use shape to bound reads

### Incomplete Feature Wiring
- F16 activations: shaders exist but wrappers don't pass dtype through
- Always trace config values end-to-end: schema -> context -> wrapper -> kernel

---

*Last updated: January 2026*
