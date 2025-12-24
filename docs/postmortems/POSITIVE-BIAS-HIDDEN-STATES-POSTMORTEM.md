# Positive Bias in Hidden States Post-Mortem (UNSOLVED)

**Status**: INVESTIGATION IN PROGRESS - "Positive bias" hypothesis DISPROVED
**Date**: 2025-12-17 23:13 UTC
**Last Updated**: 2025-12-18 00:00 UTC
**Git HEAD**: 0cff70c7
**Model**: Gemma 3 1B Q4_K_M
**Prompt**: "The color of the sky is"
**Expected Output**: "blue"
**Actual Output**: "Kaw" (garbage token 44821)

This document captures the debugging session for Gemma 3 1B producing garbage tokens despite the Q4_K quantization bug being "fixed" in a previous session. The symptoms match the earlier postmortem but occur with presumably correct weights.

---

## Symptoms

### Primary Observation
Hidden states at the **last token position** (pos=6) are **ALL POSITIVE**:
```
FINAL_HIDDEN[pos=6]: [183.6313, 42.6891, 201.0682, 63.4305, 294.5120] - ALL POSITIVE
LAST_TOKEN_HIDDEN[pos=6] after RMSNorm: [3.4719, 2.7532, 2.3350, 4.0909, 5.9923] - ALL POSITIVE
```

Meanwhile, earlier positions have **mixed signs**:
```
FINAL_HIDDEN[pos=0]: [-97.3307, -21.6574, -76.2654, -9.9215, 117.5499] - MIXED
```

### Consequence
When computing logits via dot product with embeddings:
- Tokens with more **positive embeddings** (like "Kaw") score higher
- Expected tokens like "blue" score lower despite having correct embeddings

### Logit Comparison
| Token | Token ID | Logit | Prob |
|-------|----------|-------|------|
| Kaw | 44821 | 28.35 | MAX |
| blue | 3730 | 4.81 | Low |
| sky | 7217 | ~5 | Low |
| Blue | 9595 | ~5 | Low |

The garbage token "Kaw" has the **maximum logit** due to positive bias in hidden states.

---

## Investigation Timeline

### Phase 1: Attention Variant Selection Bug
**Found**: `runAttention()` always used 'prefill' variant even during decode.
```typescript
// Bug (attention.ts:114)
const base = 'prefill';

// Fix
const base = isDecode ? 'decode' : 'prefill';
```
**Result**: Fixed but garbage output persisted.

### Phase 2: Workgroup Dispatch Bug in recordAttention
**Found**: Workgroup calculation was hardcoded to streaming-style dispatch.
```typescript
// Bug - always used streaming workgroup count
workgroups = seqLen * numHeads;

// Fix - tier-based dispatch
if (tier === 'streaming') {
  workgroups = seqLen * numHeads;
} else if (tier === 'tiled_large') {
  workgroups = Math.ceil(seqLen / 64) * numHeads;
} else {
  workgroups = Math.ceil(seqLen / 32) * numHeads;
}
```
**Result**: Fixed but garbage output persisted.

### Phase 3: Debug Readback Timing
**Found**: Debug readbacks showed zeros when using CommandRecorder (batched mode).
**Cause**: Buffers aren't populated until submit when using batch recording.
```typescript
// Added check to skip debug readback in batched mode
if (layerIdx === 0 && attnOutput instanceof GPUBuffer && !recorder) {
  // Debug readback
} else if (recorder) {
  console.log('(skipped - using batched recorder)');
}
```
**Result**: Fixed debug visibility but garbage output persisted.

### Phase 4: Streaming Attention Test
**Hypothesis**: Maybe tiled attention kernels have a bug.
**Test**: Forced streaming attention for prefill.
**Result**: Same garbage output - ruled out attention kernel as root cause.

### Phase 5: Chat Template Test
**Hypothesis**: Maybe model needs chat template formatting.
**Test**: Used `useChatTemplate: true` with Gemma 3 format.
**Result**: Same garbage output - ruled out tokenization as root cause.

### Phase 6: Position-Specific Hidden State Analysis
**Key Discovery**: Hidden state bias is position-dependent.

Added debug to read from LAST token position (which is used for next-token prediction):
```typescript
// Read hidden state at last position for logits computation
const lastTokenOffset = (numTokens - 1) * hiddenSize * 4;
```

This revealed the core issue: **positive bias accumulates specifically at the final position**.

### Phase 7: Embedding Verification
**Hypothesis**: Maybe embeddings are corrupted.
**Test**: Sampled embedding rows for "blue" (3730) and "Kaw" (44821).
```
LM_HEAD_ROW[blue]: sample=[-0.0618, 0.0417, -0.0381, 0.0574, ...]  - MIXED signs
LM_HEAD_ROW[Kaw]:  sample=[-0.0547, 0.0156, -0.0430, 0.0273, ...]  - MIXED signs
```
**Result**: Embeddings appear correct with similar magnitudes (~0.06) and mixed signs.

### Phase 8: Q4_K Dequantization Verification
**Hypothesis**: Fused matmul kernel has wrong dequant formula.
**Check**: Read `matmul_q4_fused.wgsl` kernel.
```wgsl
// Formula in kernel
w = scale * f32(q) - min_val
```
**Result**: Formula matches llama.cpp spec - should be correct.

---

## Current Hypothesis

The positive bias at the last token position suggests one of:

1. **Causal attention mask bug**: Something in how later positions attend to earlier positions
2. **Layer accumulation bug**: Residual connections accumulating positive bias over 26 layers
3. **RoPE position encoding bug**: Position-dependent values becoming biased
4. **Q4_K weights still incorrect**: Despite formula being correct, quantized weights may be wrong

### Why pos=0 has mixed signs but pos=6 is all positive

The first position (pos=0) only attends to itself (BOS token). Later positions attend to all previous positions. This suggests:
- The attention mechanism is amplifying positive values
- Or residual connections are accumulating positive bias
- Or the specific token sequence creates this bias pattern

---

## Debug Infrastructure Added

### 1. Specific Token Logit Tracking
Added debug tokens to `sampling.ts:logitsSanity()`:
```typescript
const debugTokens = [
  { id: 3730, name: 'blue' },
  { id: 77590, name: 'BLUENRG' },
  { id: 7217, name: 'sky' },
  { id: 9595, name: 'Blue' },
  { id: 51481, name: 'BLUE' },
  { id: 44821, name: 'Kaw' },     // Garbage output
  { id: 84327, name: 'Мини' },   // Russian "Mini"
  { id: 0, name: 'PAD' },
  { id: 1, name: 'BOS' },
  { id: 2, name: 'EOS' },
];
```

### 2. Last Token Position Debug
Added to `logits.ts`:
```typescript
// Read hidden state at last position (used for next-token prediction)
const lastTokenOffset = (numTokens - 1) * hiddenSize * 4;
enc.copyBufferToBuffer(hiddenStates, lastTokenOffset, staging, 0, sampleSize);
```

### 3. LM Head Row Sampling
Added to `logits.ts` to compare embedding rows:
```typescript
const debugRows = [
  { id: 3730, name: 'blue' },
  { id: 44821, name: 'Kaw' },
];
// F16 to F32 conversion for display
const f32Val = f16ToF32(f16Val);
```

### 4. Attention Variant Logging
Added to `attention.ts:recordAttention()`:
```typescript
console.log(`[ATTN] recordAttention: isDecode=${isDecode}, tier=${tier}, variant=${variant}, ...`);
```

### 5. Batched Mode Skip for Debug Readbacks
All debug readbacks now check `!recorder` to avoid reading empty buffers during batch mode.

---

## Files Modified

| File | Change |
|------|--------|
| `gpu/kernels/attention.ts` | Fixed variant selection, added logging, fixed workgroup dispatch |
| `inference/pipeline/layer.ts` | Added !recorder checks for debug readbacks |
| `inference/pipeline/sampling.ts` | Added debug tokens (Kaw, Мини, PAD, BOS, EOS) |
| `inference/pipeline/logits.ts` | Added LAST_TOKEN_HIDDEN debug, LM_HEAD_ROW sampling |
| `inference/pipeline.ts` | Changed FINAL_HIDDEN to read last position |
| `tests/benchmark/pipeline-benchmark.ts` | Added debug: true flag |

---

## Next Investigation Steps

1. **Layer-by-layer tracking at last position**: Add debug to track when positive bias starts accumulating at pos=6 specifically

2. **Compare pos=0 vs pos=6 through layers**: Are early layers already showing the divergence?

3. **Q4_K weight verification**:
   - Run quantize -> dequantize round-trip test on GPU
   - Verify actual dequantized values have negative components

4. **Reference comparison**:
   - Run same prompt through llama.cpp
   - Compare layer-by-layer hidden states

5. **Attention score visualization**:
   - Log attention weights to see if later positions have anomalous patterns

6. **Simpler model test**:
   - Try a different model (not Gemma 3) to isolate Gemma-specific issues

---

## Key Learnings (So Far)

1. **Position-specific debugging is critical**: Global buffer stats can hide position-specific issues

2. **CommandRecorder timing matters**: Debug readbacks show zeros in batched mode - add `!recorder` checks

3. **Multiple bugs can exist simultaneously**: Fixing attention variant selection revealed deeper issue

4. **Positive bias through layers is a consistent symptom**: This matches the earlier Q4_K postmortem - may be related

5. **Fast feedback loops are essential**: Quick benchmark runs with targeted grep filtering enable rapid hypothesis testing

---

## Debug Commands Used

```bash
# Quick benchmark with debug output
doppler bench inference --prompt xs --debug 2>&1 | grep -E "Layer|logits|top-5|Generated" | head -50

# Layer-specific debugging
doppler bench inference --prompt xs --debug 2>&1 | grep -E "Layer[0-9]" | head -50

# Full verbose output
doppler bench inference --prompt xs --debug --verbose 2>&1 | head -100

# Kernel selection debugging
doppler bench inference --prompt xs --debug 2>&1 | grep -E "MATMUL|variant=" | head -20
```

---

## Verification Checklist (When Fixed)

- [ ] Hidden states at last position have mixed positive/negative values
- [ ] Token "blue" has higher logit than "Kaw" for sky prompt
- [ ] Model generates "blue" or semantically correct response
- [ ] Positive bias doesn't accumulate through 26 layers
- [ ] Q4_K round-trip test preserves negative values

---

## Related Documentation

- [GEMMA3-DEBUG-POSTMORTEM.md](GEMMA3-DEBUG-POSTMORTEM.md) - Original Q4_K quantization fix
- [DOPPLER-TROUBLESHOOTING.md](../DOPPLER-TROUBLESHOOTING.md) - Debug guide
- [DEBUG_SESSION.md](../DEBUG_SESSION.md) - Quick start for debugging

---

## UPDATE: "Positive Bias" Hypothesis DISPROVED (2025-12-18)

### New Evidence

With improved debug tooling (CommandRecorder disabled in debug mode), we now see:

```
LAYER_0_LAST[pos=6]:  min=-10.053, max=9.080   (MIXED)
LAYER_12_LAST[pos=6]: min=-136.646, max=70.206 (MIXED)
LAYER_25_LAST[pos=6]: min=-757.428, max=1421.781 (MIXED)
```

**Hidden states have MIXED SIGNS throughout all 26 layers.** The earlier "all positive" observation was a sampling artifact from reading only 5 values.

### Tests Performed

| Test | Result | Implication |
|------|--------|-------------|
| Q4_K GPU dequant correctness | **PASSED** | Dequantization preserves negative values |
| Layer-by-layer min/max at last position | **MIXED** | No positive bias accumulation |
| Gemma chat template | **APPLIED** | Correct `<start_of_turn>user...<end_of_turn>` format |
| Model output | **GARBAGE** | "lim" (49%), "Kaw" (84%) instead of "blue" |

### What This Means

1. **Q4_K dequantization is NOT the root cause**
2. **Chat template is NOT the root cause**
3. **Positive bias is NOT the root cause**
4. **Something else is fundamentally wrong**

### New Hypotheses

1. **Model weights corrupted during conversion** (50%) - The .rdrr files may have been incorrectly converted
2. **Attention kernel bug** (20%) - Despite mixed signs, attention may be computing wrong values
3. **RoPE or position encoding bug** (15%) - Position-dependent errors
4. **Tokenizer mismatch** (10%) - Vocabulary or special token handling wrong
5. **Buffer lifecycle bug** (5%) - Use-after-free or wrong buffer being read

### Recommended Next Steps

1. **Compare against llama.cpp reference** - Run same model with same prompt, compare logits
2. **Verify model conversion** - Check if .gguf → .rdrr conversion preserved weights correctly
3. **Test with different model** - Try a different model (not Gemma 3) to isolate Gemma-specific issues
4. **Add attention output verification** - Compare Q*K^T attention scores against reference

---

## Instructions for Next Agent

### Quick Start
```bash
# 1. Reproduce the issue
doppler bench inference --prompt xs --debug 2>&1 | grep -E "FINAL_HIDDEN|LAST_TOKEN|blue|Kaw|Generated"

# 2. Verify the symptom: hidden states at last position should be ALL POSITIVE
# Look for: FINAL_HIDDEN[pos=6]: [183.x, 42.x, 201.x, ...] - all positive numbers
```

### Priority Investigation Path

1. **Verify Q4_K dequantization actually produces negative values**:
   - Create a test that loads a weight block, runs it through the GPU matmul_q4_fused kernel, and verifies negatives exist
   - File: `gpu/kernels/matmul_q4_fused.wgsl`
   - The formula `w = scale * f32(q) - min_val` should produce negatives when min_val is positive

2. **Layer-by-layer hidden state tracking at last position**:
   - Add logging to track hidden states at pos=N-1 (last token) after each layer
   - Find which layer the positive bias starts appearing
   - Key file: `inference/pipeline/layer.ts`

3. **Compare against llama.cpp reference**:
   ```bash
   # Run llama.cpp with same prompt
   ./main -m gemma-3-1b-q4_K_M.gguf -p "<bos>The color of the sky is" -n 1 --verbose
   # Compare hidden state statistics
   ```

4. **Check if weights are actually quantized correctly**:
   - The Q4_K quantization bug was "fixed" but may have regressed or never been fully fixed
   - Test: quantize a small tensor with known negatives, dequantize on GPU, verify negatives exist

### Key Files to Instrument

| File | What to Add |
|------|-------------|
| `inference/pipeline/layer.ts:processLayerGPU` | Log hidden state stats at LAST token position after each sub-step |
| `gpu/kernels/matmul_q4_fused.wgsl` | Add debug to output dequantized weight samples |
| `inference/pipeline/logits.ts` | Already has LAST_TOKEN_HIDDEN debug - use it |

### Hypothesis Ranking (Most to Least Likely)

1. **Q4_K weights still wrong** (70%): The "fixed" quantization may not actually be producing correct values
2. **Causal mask / position bug** (15%): Something in how attention handles position N-1
3. **Residual accumulation bug** (10%): Subtle sign flip in residual add or norm
4. **RoPE position encoding** (5%): Position-dependent values going wrong

### What's Already Been Ruled Out

- Attention kernel variant selection (fixed)
- Workgroup dispatch calculation (fixed)
- Debug readback timing with CommandRecorder (fixed)
- Streaming vs tiled attention (tested - same result)
- Chat template formatting (tested - same result)
- Embedding row values for blue/Kaw (verified correct)

### Success Criteria

When fixed, this command should output "blue" or a semantically correct color word:
```bash
doppler bench inference --prompt xs --debug 2>&1 | grep "Generated"
# Expected: Generated: "blue" or "Blue" or similar
# Current:  Generated: "Kaw" (garbage)
```

And hidden states at last position should have mixed signs:
```bash
doppler bench inference --prompt xs --debug 2>&1 | grep "FINAL_HIDDEN"
# Expected: FINAL_HIDDEN[pos=6]: [-50.x, 30.x, -20.x, ...] - MIXED signs
# Current:  FINAL_HIDDEN[pos=6]: [183.x, 42.x, 201.x, ...] - ALL POSITIVE
```

---

*Investigation started: 2025-12-17 ~20:00 UTC*
*Last updated: 2025-12-17 23:13 UTC*
*Status: UNSOLVED - awaiting next debug session*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

