# Gemma 3 Hidden State Magnitude Gap Post-Mortem

**Status**: IN PROGRESS - GPU sync bug fixed, 2.6x gap remaining
**Date**: 2025-12-21
**Model**: Gemma 3 1B IT Q4_K_M
**Prompt**: "The color of the sky is"
**Expected Output**: "blue" or similar
**Actual Output**: "攜 transcript claims\\# फी\\# color انہ" (garbage)

---

## Key Discovery: GPU Sync Bug in Embedding Scaling

**ROOT CAUSE IDENTIFIED AND FIXED**: `scaleGPUBuffer()` in `embed.js` was not waiting for GPU completion before returning.

```typescript
// BEFORE (BROKEN):
device.queue.submit([encoder.finish()]);
uniformBuffer.destroy();
return outputBuffer;  // Returns immediately - caller reads stale data!

// AFTER (FIXED):
device.queue.submit([encoder.finish()]);

// CRITICAL: Wait for GPU to complete scale operation before returning
await device.queue.onSubmittedWorkDone();

uniformBuffer.destroy();
return outputBuffer;
```

**Impact**: This fix improved Layer 0 hidden state magnitude from ~24 to ~86 (3.5x improvement).

---

## Current State After Fix

| Layer | HuggingFace | DOPPLER (Before Fix) | DOPPLER (After Fix) | Gap |
|-------|-------------|---------------------|---------------------|-----|
| After Layer 0 | 227 | 24.47 | 86.41 | 2.6x smaller |
| After Layer 25 | 15,168 | 377.78 | 1,174.48 | 13x smaller |
| Final Output | "blue" | garbage | garbage | Still broken |

**The remaining 2.6x gap at Layer 0 is the current investigation target.**

---

## What Was Fixed This Session

### 1. GPU Sync Bug in `scaleGPUBuffer` (embed.js:142)
- **Symptom**: `embed()` returned maxAbs=25.19 but pipeline saw maxAbs=1.27 for the same buffer
- **Cause**: Missing `await device.queue.onSubmittedWorkDone()` after submit
- **Fix**: Added await to ensure GPU completes scaling before returning

### 2. GELU Activation Default for Gemma 3 (config.js:343)
- **Issue**: `normalizeActivation()` defaulted to 'silu' if manifest didn't specify activation
- **Fix**: Added Gemma 3 specific default: `config.hidden_activation ?? (isGemma3 ? 'gelu' : undefined)`
- **Note**: This fix didn't change output - the manifest may already specify GELU

---

## Investigation Attempted But Not Completed

### Debug Logging in Attention Function
Added debug logging to `runLayerAttentionGPU` to trace:
- Normalized input magnitude after `input_layernorm`
- Q/K/V projection output magnitudes

**Issue**: The debug logging is not appearing in output even though:
- `hasRecorder=false` is confirmed (batching disabled in debug mode)
- Layer 0 `attnOutput` shows as GPUBuffer
- Other layer debug messages appear normally

**Possible causes**:
1. The function may be taking a different code path
2. TypeScript compilation issue
3. The function `runLayerAttentionGPU` may not be called directly

**Current debug code added** (not yet producing output):
```typescript
// In runLayerAttentionGPU, at top of function:
if (config.layerIdx === 0) {
  console.log(`[runLayerAttentionGPU] CALLED for layer 0, debug=${debug}`);
}

// Before Q/K/V projections:
if (layerIdx === 0 && debug) {
  console.log(`[Attn L0] DEBUG ENTRY: ...`);
  // GPU buffer readback for normed input
  // GPU buffer readback for Q/K/V outputs
}
```

---

## Remaining Hypotheses

1. **Q4K dequantization produces smaller values** (40%)
   - Matmul with Q4K weights may have incorrect scale factors
   - Need to compare dequantized weight values against reference

2. **Attention scaling factor wrong** (30%)
   - Scale = 1/sqrt(headDim) may not be applied correctly
   - Softmax may be over-normalizing

3. **Per-layer growth factor issue** (20%)
   - DOPPLER: Layer 0 input (86) → Layer 0 output (86) = 1x growth
   - HuggingFace: Layer 0 input (~25 scaled) → Layer 0 output (227) = 9x growth
   - Something in attention or FFN is not amplifying correctly

4. **Residual connection order** (10%)
   - Gemma 3 sandwich norm: norm → op → residual + norm(op_out)
   - May need to verify residual is adding to correct values

---

## Files Modified This Session

| File | Change |
|------|--------|
| `inference/pipeline/embed.js:142` | Added GPU sync after scale operation |
| `inference/pipeline/config.js:343` | Added GELU default for Gemma 3 |
| `inference/pipeline/attention.js` | Added debug logging (not yet working) |

---

## Commands for Next Session

```bash
# Rebuild after changes
npm run build:doppler

# Test inference with debug
npm run doppler -- bench inference --model gemma-3-1b-it-q4 --prompt xs --debug

# Check if attention debug is working
npm run doppler -- bench inference --model gemma-3-1b-it-q4 --prompt xs --debug 2>&1 | grep -E "runLayerAttentionGPU|Attn L0"

# Verify Llama still works (should output "blue")
npm run doppler -- bench inference --model llama-3.2-1b-instruct-q4 --prompt xs --debug
```

---

## Next Steps

1. **Debug why `runLayerAttentionGPU` logging isn't appearing**
   - Check if `recordLayerAttentionGPU` is being called instead
   - Verify the `doAttention` wrapper is passing debug flag correctly
   - Try adding debug logging to `recordLayerAttentionGPU` as well

2. **Compare Q/K/V magnitudes with HuggingFace**
   - Once logging works, compare Q, K, V values layer-by-layer
   - Check if Q4K dequantization is producing correct magnitudes

3. **Check attention output projection**
   - O_proj may be scaling down incorrectly
   - Compare attention output magnitude with reference

4. **Verify FFN gate/up projections**
   - GELU activation may have correct formula but wrong input magnitude
   - Check if FFN is amplifying correctly

---

## Reference: HuggingFace Hidden State Values

```python
# Run this to get reference values:
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('google/gemma-3-1b-it', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')
inputs = tokenizer('The color of the sky is', return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
for i, h in enumerate(outputs.hidden_states):
    print(f'Layer {i}: maxAbs={h[0,-1,:].float().abs().max().item():.2f}')
```

Expected output:
```
Layer 0: maxAbs=227.00  # After layer 0
Layer 16: maxAbs=8384.00
Layer 25: maxAbs=15168.00  # Before final norm
```

---

## Related Documentation

- [2025-12-18-hidden-state-underaccumulation.md](2025-12-18-hidden-state-underaccumulation.md) - Previous investigation
- [2025-12-17-positive-bias-hidden-states.md](2025-12-17-positive-bias-hidden-states.md) - Earlier hypothesis (disproved)
- [DOPPLER-TROUBLESHOOTING.md](../OPERATIONS.md) - Debug guide
