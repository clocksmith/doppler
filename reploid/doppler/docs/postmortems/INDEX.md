# DOPPLER Post-Mortems Index

Quick reference for debugging history and lessons learned.

---

## Summary Table

| Post-Mortem | Date | Status | Root Cause |
|-------------|------|--------|------------|
| [Gemma 3 Magnitude Gap](#gemma3-magnitude-gap) | Dec 2025 | IN PROGRESS | GPU sync bug fixed, 2.6x gap remaining |
| [Gemma 3 Q4K Garbage Output](#gemma3-1b-q4k-garbage-output) | Dec 2025 | RESOLVED | Q4K layout mismatch + q_norm offset |
| [Hidden State Under-Accumulation](#hidden-state-under-accumulation) | Dec 2025 | SUPERSEDED | Merged into Garbage Output PM |
| [Positive Bias Hidden States](#positive-bias-hidden-states) | Dec 2025 | DISPROVED | Sampling artifact, not real issue |
| [Softmax Uniform Buffer](#softmax-uniform-buffer) | Dec 2025 | RESOLVED | Swapped innerSize/outerSize |
| [Pipeline Verification](#pipeline-verification) | Dec 2025 | RESOLVED | Identified FFN explosion |
| [Gemma 3 Debug](#gemma3-debug) | Dec 2025 | RESOLVED | Q4K quantization format mismatch |
| [MoE Explicit Layout](#moe-explicit-layout) | Dec 2025 | RESOLVED | WebGPU 'auto' layout binding mismatch |
| [BF16 2D Dispatch](#bf16-2d-dispatch) | Dec 2025 | RESOLVED | 2D dispatch without linearization |

---

## Post-Mortem Details

### Gemma3-Magnitude-Gap

**Status:** IN PROGRESS | **File:** [GEMMA3-MAGNITUDE-GAP-2025-12-21.md](GEMMA3-MAGNITUDE-GAP-2025-12-21.md)

Found and fixed GPU sync bug in `scaleGPUBuffer()` - function returned before GPU completed scaling operation, causing pipeline to read stale data. This improved Layer 0 magnitude from 24 to 86 (3.5x improvement). However, a 2.6x gap remains vs HuggingFace reference (86 vs 227 at Layer 0). Added GELU activation default for Gemma 3. Debug logging added to attention function but not yet appearing in output - investigation paused. Next: determine why `runLayerAttentionGPU` debug logs aren't appearing, then trace Q/K/V magnitudes.

---

### Gemma3-1B-Q4K-GARBAGE-OUTPUT

**Status:** RESOLVED | **File:** [GEMMA3-1B-Q4K-GARBAGE-OUTPUT-2025-12-18.md](GEMMA3-1B-Q4K-GARBAGE-OUTPUT-2025-12-18.md)

Q4K weights stored in packed 256-block stream incompatible with DOPPLER's row-wise fused matmul addressing. Matrices with K not divisible by 256 (Gemma: K=1152) had corrupted Q/K/V projections. Secondary issue: missing `(1+weight)` offset for q_norm/k_norm. Debug logging artifact under-reported hidden state magnitudes. Fix: loader fallback dequantizes packed Q4K to F16 for correctness. Model now outputs "blue" correctly with HuggingFace-matching hidden state magnitudes.

---

### Hidden-State-Under-Accumulation

**Status:** SUPERSEDED | **File:** [HIDDEN-STATE-UNDERACCUMULATION-2025-12-18.md](HIDDEN-STATE-UNDERACCUMULATION-2025-12-18.md)

Observed hidden states 10-30x smaller than HuggingFace reference. Investigation identified embedding was correct but layer 0 output was 9x smaller than expected. Hypothesized Q4K dequantization scale factors were wrong. This postmortem was superseded by GEMMA3-1B-Q4K-GARBAGE-OUTPUT which identified the actual root cause: Q4K layout mismatch causing fallback to dequantized weights, plus debug sampling artifact that under-reported true magnitudes. The "under-accumulation" was partially measurement error.

---

### Positive-Bias-Hidden-States

**Status:** DISPROVED | **File:** [POSITIVE-BIAS-HIDDEN-STATES-POSTMORTEM.md](POSITIVE-BIAS-HIDDEN-STATES-POSTMORTEM.md)

Initially observed all-positive hidden states at last token position. Extensive investigation fixed attention variant selection bug, workgroup dispatch bug, and debug readback timing issues. Later discovered the "positive bias" was a sampling artifact - debug only read 5 values, not full 1152-dim vector. Full vector shows mixed positive/negative signs throughout all 26 layers. Key learning: position-specific debugging critical; global buffer stats hide position-specific issues. Hypothesis was wrong but investigation improved debug infrastructure.

---

### Softmax-Uniform-Buffer

**Status:** RESOLVED | **File:** [SOFTMAX-UNIFORM-BUFFER-POSTMORTEM.md](SOFTMAX-UNIFORM-BUFFER-POSTMORTEM.md)

Softmax kernel failed correctness tests with maxError=0.137 (should be <1e-5). Root cause: TypeScript wrote `batchSize` at offset 0, `inferredSize` at offset 4, but WGSL expected `innerSize` at offset 0, `outerSize` at offset 4. Swapped dimensions caused wrong workgroup dispatch and misaligned row boundaries. Fix: corrected uniform buffer writes in `runSoftmax` and `recordSoftmax`. Note: attention uses inline softmax, so this didn't affect main inference. Lesson: uniform buffer layout is error-prone; add comments documenting WGSL struct layout.

---

### Pipeline-Verification

**Status:** RESOLVED | **File:** [PIPELINE-VERIFICATION-POSTMORTEM.md](PIPELINE-VERIFICATION-POSTMORTEM.md)

Systematic verification identified FFN down projection explosion (min=-3078, max=756) causing near-uniform logits (~3% top token confidence) and garbage output (Telugu, Kannada, Japanese scripts). Post-FFN sandwich norm masked the explosion by normalizing values. Verified embedding, gather, scaling, Q4K dequant were correct. The explosion was later traced to Q4K quantization format mismatch (see GEMMA3-DEBUG-POSTMORTEM). Lesson: sandwich norms can mask issues; track values through full pipeline; near-uniform logits indicate corruption.

---

### Gemma3-Debug

**Status:** RESOLVED | **File:** [GEMMA3-DEBUG-POSTMORTEM.md](GEMMA3-DEBUG-POSTMORTEM.md)

Model output `<unused16>` tokens instead of coherent text. Hidden states showed positive bias accumulation (all values positive when negatives expected). Root cause: quantizer produced data in wrong format - used `q * scale + min` instead of llama.cpp's `d * sc * q - dmin * min`. Negative weights dequantized as positive, accumulating bias through 26 layers. Fix: rewrote `quantizeQ4KBlock()` to match llama.cpp byte layout. Lesson: format compatibility matters; "close enough" isn't good enough for quantization.

---

### MoE-Explicit-Layout

**Status:** RESOLVED | **File:** [MOE-EXPLICIT-LAYOUT-POSTMORTEM.md](MOE-EXPLICIT-LAYOUT-POSTMORTEM.md)

MoE gather kernel compiled successfully but didn't execute - `tokenCounts` array was all zeros. Root cause: shader has 6 bindings but `count_and_map` entry point only uses 4. WebGPU's `layout: 'auto'` creates layout with only used bindings, causing silent mismatch when creating bind group with all 6 entries. No validation error thrown. Fix: created explicit bind group layout with all 6 bindings. Lesson: avoid 'auto' layout for multi-entry-point shaders; silent WebGPU failures are dangerous.

---

### BF16-2D-Dispatch

**Status:** RESOLVED | **File:** [BF16-2D-DISPATCH-POSTMORTEM.md](BF16-2D-DISPATCH-POSTMORTEM.md)

Large vocab models (Gemma 262K, Mistral 32K) produced garbage output. Embeddings for token IDs >8192 returned zeros. Root cause: BF16->F32 kernel used 2D dispatch for tensors exceeding 65535 workgroups, but kernel only used `global_id.x`, ignoring `global_id.y`. Only first ~33M elements converted. Fix: compute linear index from 2D dispatch using `workgroupsX` uniform. Lesson: always linearize 2D dispatch; large vocab models expose bugs; pass dispatch info to kernels for correct linearization.

---

## Common Patterns

### Silent WebGPU Failures
- MoE explicit layout: no error, kernel just didn't run
- BF16 2D dispatch: no error, partial data processed

### Quantization Format Bugs
- Q4K packed vs row-wise layout
- Scale/min encoding mismatches
- Sign handling in dequantization

### Uniform Buffer Mismatches
- TypeScript/WGSL struct field order
- Always add comments documenting expected layout

### Debug Instrumentation Errors
- Sampling only first N values hides real distribution
- Command batching affects readback timing
- Position-specific values differ from global stats

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

