# Gemma 3 1B Q4K Inference Garbage Output Post-Mortem

**Status**: RESOLVED ★  
**Date**: 2025-12-18  
**Model**: `doppler/models/gemma-3-1b-it-q4/` (`Q4_K_M`, `dcc83ea8…`)  
**Prompt**: `The color of the sky is`  
**Expected**: First token `blue` (xs prompt sanity check)  
**Actual (broken)**: Garbage tokens (mixed scripts), unstable logits

---

## Summary

Two issues compounded into one confusing failure:

1. **Real correctness bug**: Q4_K_M weights were stored in a “packed” 256-value block stream that is incompatible with DOPPLER’s **row-wise** Q4K fused matmul addressing. This corrupted major projections (notably Q/K/V) for matrices where `K` is not divisible by 256 (Gemma 3 uses `K=1152` heavily).
2. **Measurement + harness confusion**: Debug logging only sampled small prefixes of the hidden vector (under-reporting `maxAbs` by 10–30×), and the benchmark harness always applied the Gemma chat template (changing the expected next token).

After fixes, DOPPLER:

- Produces `blue` for the `xs` prompt in the benchmark harness.
- Matches HuggingFace hidden-state magnitudes at key checkpoints (Layer 0 ~227, Layer 16 ~8k, final-norm output ~60).

---

## Impact

- Gemma 3 1B IT Q4K was unusable for browser inference (garbage output).
- Debugging was misdirected by misleading hidden-state “underaccumulation” metrics and prompt-mode mismatch.

---

## Timeline

| Time (UTC) | Event |
|-----------:|-------|
| 2025-12-18 | Reproduced garbage output on `xs` prompt; began layer-by-layer tracing. |
| 2025-12-18 | Verified several suspected components were correct (BF16 norm weights, residual add, GELU, +1 applied once). |
| 2025-12-18 | Observed “hidden-state underaccumulation” vs HuggingFace based on debug logs; prioritized scaling/matmul/norm hypotheses. |
| 2025-12-18 | Identified Q4K layout incompatibility by comparing tensor byte size against expected row-wise Q4K size for `[rows, K]` weights. |
| 2025-12-18 | Implemented loader fallback (packed Q4K → dequantized `f16`) to restore correctness without reconversion. |
| 2025-12-18 | Corrected Gemma 3 `q_norm/k_norm` handling to use `(1 + weight)` like other RMSNorms. |
| 2025-12-18 | Fixed benchmark prompt-mode confusion and corrected debug stats to read full last-token vectors. |
| 2025-12-18 | Verified output `blue` and HuggingFace-matching magnitudes at checkpoints. |

---

## Symptoms

- Output tokens were incoherent (often non-English scripts).
- Hidden-state stats appeared **too small** vs HuggingFace (reported 10–30× smaller).
- Per-layer behavior looked “stable but wrong”, leading to incorrect hypotheses (embedding scaling, RMSNorm divisor, Q4K magnitude).

---

## Root Causes

### 1) Q4K weight layout mismatch (packed vs row-wise)

**What happened**

- The Q4K quantizer/converter wrote blocks as a flat stream:
  - `numBlocks = ceil(numElements / 256)`
  - blocks are taken sequentially across the entire tensor.
- The fused Q4K matmul kernel assumes row-wise block layout:
  - `rowBlocks = ceil(K / 256)`
  - block offset = `row * rowBlocks + blockWithinRow`
- When `K` is not divisible by 256 (Gemma 3: `K=1152`, `rowBlocks=5`), blocks in the packed stream cross row boundaries. The kernel then reads the wrong block metadata (scales/mins/nibbles) for most rows.

**Why this produced garbage**

Q/K/V projections dominate early-layer signal. Once they are corrupted, attention and residual stream accumulate nonsense, producing garbage logits even if RMSNorm, GELU, residual kernels, and BF16 weights are otherwise correct.

### 2) q_norm / k_norm weight offset mismatch

Gemma 3 uses `Gemma3RMSNorm`, which applies:

```
output = rmsnorm(x) * (1 + weight)
```

This applies to layer norms and the per-head `q_norm`/`k_norm` modules. Missing the `+1` offset shifts Q/K distributions and changes attention behavior.

### 3) Benchmark prompt-mode mismatch (chat template vs raw)

The benchmark harness always enabled `useChatTemplate: true` for Gemma 3 IT. HuggingFace’s next-token behavior differs between:

- raw prompt: `"The color of the sky is"`
- chat template: `"<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n"`

This caused “expected blue” checks to be performed against a different prompt mode than intended.

### 4) Hidden-state “underaccumulation” was a debug sampling artifact

Per-layer debug logs computed `maxAbs` from only the first ~64 floats of the last-token vector. For Gemma 3, the true `maxAbs` often occurs later in the 1152-dim vector, so the logged values were systematically low.

☡ This is a classic “instrumentation bug”: it looked like a model-scale issue but was primarily a measurement error.

---

## Fixes Implemented

### A) Loader correctness fallback for packed Q4K weights

- Detect incompatible Q4K storage for 2D tensors by comparing expected row-wise byte size:
  - `expected = rows * ceil(cols/256) * 144`
  - if stored `size < expected`, treat as packed/incompatible
- For packed weights, bypass fused Q4K matmul by dequantizing to `f16` and using standard matmul.

This restores correctness for already-converted models without requiring reconversion.

### B) Apply `(1 + weight)` for `q_norm` and `k_norm`

- Load `q_norm.weight` and `k_norm.weight` using the same +1 offset path as other Gemma 3 norms.

### C) Make benchmark chat templating explicit

- Added `useChatTemplate` to benchmark config, default `false`.
- Token counting in results now reflects the exact prompt text passed into tokenization (raw vs chat-templated).

### D) Fix hidden-state debug stats to use full vectors

- Layer/logits debug readbacks now compute stats over the full last-token hidden vector (1152 floats), not a prefix sample.

---

## Verification

### DOPPLER (after fixes)

Command:

```bash
npm run doppler -- bench inference --prompt xs --max-tokens 1 --retries 0 --quiet
```

Expected behavior:

- Prefill top-5 includes `blue`
- First sampled token is `blue`

Key checkpoint magnitudes (raw prompt, `seq_len=7`):

| Stage | HuggingFace (BF16) | DOPPLER (Q4K) |
|------:|---------------------:|--------------:|
| After layer 0 (last token maxAbs) | ~227 | ~227 |
| After layer 16 (last token maxAbs) | ~8384 | ~8193 |
| After final norm (last token maxAbs) | ~60 | ~59 |

### HuggingFace reference (raw prompt)

Verified with `transformers` local snapshot `dcc83ea8…` using `output_hidden_states=True` and reading the last token’s `maxAbs` at:

- embeddings output
- after layer 0 (hidden_states[1])
- after layer 16 (hidden_states[17])
- after final norm (hidden_states[26])

---

## Preventative Actions

1. **Version Q4K tensor layout in the manifest**: record whether tensors are `rowwise_q4k` vs `packed_q4k` so kernels can reject incompatible layouts early.
2. **Quantizer fix**: for 2D matmul weights, quantize per-row with per-row padding to 256-aligned blocks to match fused matmul addressing.
3. **Add regression tests**:
   - Q4K matmul tests where `K` is not divisible by 256 (Gemma-like sizes, e.g. `K=1152`).
   - Loader test that asserts row-wise byte-size invariants for Q4K 2D weights.
4. **Debug instrumentation rules**: any “maxAbs” metric must specify whether it is sampled or full-vector. Prefer full-vector for correctness debugging.
5. **Benchmark prompt contract**: record whether chat template was applied and the exact token IDs of the prompt in benchmark output.

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes and the OPFS purge helper.

