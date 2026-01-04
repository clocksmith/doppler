# Gemma 3 q_norm/k_norm +1 Offset Fix Post-Mortem

**Status**: RESOLVED
**Date**: 2025-12-25
**Model**: Gemma 3 1B IT Q4_K_M
**Prompt**: "The color of the sky is"
**Expected Output**: "blue" or similar coherent text
**Actual Output (before fix)**: " объ khác reino kolor usługi کرن编译器 remembr" (garbage)
**Actual Output (after fix)**: " blue, the blue, the color of" (correct)

---

## Root Cause

**All Gemma 3 RMSNorm weights use `(1 + weight)` formula, including q_norm and k_norm.**

The loader and attention pipeline were NOT applying the +1 offset to q_norm/k_norm weights, while correctly applying it to all other norm weights (input_layernorm, post_attention_layernorm, etc.).

### HuggingFace Reference (modeling_gemma3.py)

```python
class Gemma3RMSNorm(nn.Module):
    def forward(self, x):
        # ALL Gemma 3 norms use (1 + weight)
        return x * torch.rsqrt(variance + self.eps) * (1 + self.weight)
```

Both `q_norm` and `k_norm` in `Gemma3Attention` are instances of `Gemma3RMSNorm`, confirming they require the +1 offset.

---

## The Contradiction

Prior documentation was inconsistent:

| Source | Claim |
|--------|-------|
| `2025-12-18-gemma3-q4k-garbage-output.md` | q_norm/k_norm SHOULD have +1 offset |
| `docs/TODO.md` (before fix) | q_norm/k_norm should NOT have +1 offset |
| Current code (before fix) | q_norm/k_norm did NOT have +1 offset |

**Resolution**: HuggingFace source code is authoritative. ALL Gemma 3 norms use `(1 + weight)`.

---

## Files Modified

### 1. loader/doppler-loader.ts (lines 1458-1462)

```typescript
// BEFORE (BROKEN):
tryLoad(['self_attn.q_norm.weight', 'attn_q_norm.weight']),
tryLoad(['self_attn.k_norm.weight', 'attn_k_norm.weight']),

// AFTER (FIXED):
// Gemma 3: q_norm and k_norm use Gemma3RMSNorm with (1+weight) formula
// Same as layer norms - all Gemma 3 norms use (1+weight)
tryLoadNorm(['self_attn.q_norm.weight', 'attn_q_norm.weight']),
tryLoadNorm(['self_attn.k_norm.weight', 'attn_k_norm.weight']),
```

### 2. inference/pipeline/attention.ts (lines 210-214, 230-231)

GPU path - changed `getWeightBuffer` to `getNormWeightBuffer`:

```typescript
// BEFORE:
if (hasQNorm && layerWeights.qNorm) {
  const qNormBuf = getWeightBuffer(layerWeights.qNorm, 'q_norm');

// AFTER:
if (hasQNorm && getNormWeightBuffer && layerWeights.qNorm) {
  const qNormBuf = getNormWeightBuffer(layerWeights.qNorm, 'q_norm');
```

### 3. inference/pipeline/attention.ts (lines 453-456, 469-470)

Recorded path - same change for batched execution:

```typescript
// BEFORE:
if (layerWeights.qNorm) {
  const qNormBuf = getWeightBuffer(layerWeights.qNorm, 'q_norm');

// AFTER:
if (layerWeights.qNorm && getNormWeightBuffer) {
  const qNormBuf = getNormWeightBuffer(layerWeights.qNorm, 'q_norm');
```

### 4. docs/TODO.md (lines 26-28)

Updated to reflect correct behavior:

```markdown
### Recent Fix: q_norm/k_norm +1 Offset Correction
Gemma 3's per-head Q/K normalizations use Gemma3RMSNorm with (1+weight) formula,
SAME as layer norms. These weights MUST have +1 offset applied.
```

---

## Why OPFS Cache Clear + Reconversion Was Required

The +1 offset for norm weights can be applied at two stages:

1. **Conversion time**: Bake the +1 into the stored weight values
2. **Runtime**: Apply +1 when loading weights into GPU buffers

DOPPLER uses a hybrid approach:
- Loader's `tryLoadNorm()` applies +1 to Float32Array weights
- Pipeline's `getNormWeightBuffer()` handles GPU buffer offset if needed

Since the model was previously converted with `tryLoad()` for q_norm/k_norm (no +1 offset), the stored weights in OPFS were incorrect. Simply fixing the runtime code wasn't enough - the model needed reconversion to bake in the correct +1 offset.

**Reconversion command**:
```bash
npx tsx tools/convert-cli.ts \
  ~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752/ \
  models/gemma-3-1b-it-q4 \
  --quantize q4_k_m \
  --model-id gemma-3-1b-it-q4 \
  --verbose
```

---

## Verification

### Before Fix
```
"generated_text": " объ khác reino kolor usługi کرن编译器 remembr"
```

### After Fix
```
"generated_text": " blue, the blue, the color of"
```

Performance unchanged at 13-18 tok/s decode, ~750ms TTFT.

---

## Lessons Learned

1. **Authoritative source**: When documentation conflicts, check the original implementation (HuggingFace transformers source code)

2. **Gemma 3 is special**: Unlike Llama which has `rmsNormWeightOffset = false`, ALL Gemma 3 norms use the `(1 + weight)` formula - not just layer norms but also per-head Q/K norms

3. **OPFS cache invalidation**: When fixing weight processing bugs, the browser cache retains old weights. Users must clear OPFS before the fix takes effect

4. **Norm weight offset is subtle**: The +1 offset changes weight values from ~0 (centered around 0) to ~1 (centered around 1). Missing this completely breaks normalization scaling

---

## Related Documentation

- [2025-12-18-gemma3-q4k-garbage-output.md](2025-12-18-gemma3-q4k-garbage-output.md) - Previous investigation that identified q_norm/k_norm as needing offset
- [2025-12-21-gemma3-magnitude-gap.md](2025-12-21-gemma3-magnitude-gap.md) - Investigation of hidden state magnitude issues (now superseded by this fix)
- [HuggingFace modeling_gemma3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py) - Authoritative reference
