# Hidden State Under-Accumulation Post-Mortem

**Status**: UNSOLVED - Root cause identified but not fixed
**Date**: 2025-12-18
**Model**: Gemma 3 1B IT Q4_K_M (converted from safetensors)
**Prompt**: "The color of the sky is"
**Expected Output**: "blue" or similar
**Actual Output**: "postwarabisavits Stevens..." (garbage)

---

## Key Discovery

**DOPPLER hidden states are 10-30x SMALLER than HuggingFace reference:**

| Layer | HuggingFace | DOPPLER | Ratio |
|-------|-------------|---------|-------|
| After embedding | 19.25 | ~19 | ~1x (OK) |
| After layer 0 | 227.00 | 24.47 | HF 9x larger |
| After layer 16 | 8,384 | 316.91 | HF 26x larger |
| Before final norm | 15,168 | 821 | HF 18x larger |
| After final norm | 60.25 | ? | HF normalizes down |

**The issue is UNDER-accumulation through residual connections, not explosion.**

Previous hypothesis about "explosion" was WRONG - comparing maxAbs=821 to "expected <100" was based on incorrect assumption. HuggingFace actually produces maxAbs=15,000+ before final normalization!

---

## Verified CORRECT

1. **BF16 norm weights match exactly** after safetensors→RDRR conversion
   - Original: `[4.09375, 4.375, 2.875, 4.0, 4.34375]`
   - RDRR: `[4.09375, 4.375, 2.875, 4.0, 4.34375]` (identical)

2. **+1 norm weight offset applied correctly** (not duplicated)
   - Loader applies offset during load
   - Pipeline skips if already GPUBuffer
   - Final norm weights: `[6.1563, 21.0, 3.7813, 21.0, 6.6250]` = original + 1

3. **Residual add kernel is correct**
   - WGSL: `output[idx] = a[idx] + b[idx]`

4. **GELU activation uses correct tanh approximation**
   - Coefficient 0.044715 matches PyTorch's gelu_pytorch_tanh

5. **Model naming standardized** to `gemma-3-1b-it-q4`

---

## Investigation Timeline

### Session 1: "Positive Bias" Hypothesis (DISPROVED)
- Thought hidden states were all positive at last position
- Discovered this was sampling artifact from reading only 5 values
- Full buffer shows mixed positive/negative signs

### Session 2: "Hidden State Explosion" Hypothesis (DISPROVED)
- Thought maxAbs=821 was "too large"
- Discovered HuggingFace produces maxAbs=15,000+ normally
- DOPPLER values are actually 10-30x TOO SMALL

### Session 3: Current Discovery
- Hidden states under-accumulate through residual connections
- Divergence starts from Layer 0 output (9x smaller than HF)
- Problem compounds through layers

---

## Likely Root Causes (Prioritized)

1. **Embedding scaling issue** (40%)
   - HF raw embedding: maxAbs=19.25
   - HF scaled (×√1152): maxAbs=653
   - Need to verify DOPPLER embedding scaling

2. **Attention output magnitude wrong** (30%)
   - Attention may be producing too-small outputs
   - Q4K dequantization during attention matmul may be wrong
   - Softmax may be over-normalizing

3. **FFN output magnitude wrong** (20%)
   - Gate/up projection producing small values
   - Down projection scaling issue

4. **RMSNorm scaling wrong** (10%)
   - Normalization may be dividing by wrong RMS value

---

## What Was NOT Tried

1. **Detailed attention output comparison** - Compare Q, K, V, attention scores, output layer-by-layer
2. **Q4K dequantization verification** - Verify actual dequantized values during matmul
3. **Embedding layer debug** - Verify embedding lookup produces correct values
4. **FFN intermediate values** - Compare gate, up, activation, down outputs

---

## Reproduction Commands

```bash
# Run inference with debug
npm run doppler -- bench inference --prompt xs --debug

# Compare with HuggingFace reference
python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('google/gemma-3-1b-it', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')
inputs = tokenizer('The color of the sky is', return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
for i, h in enumerate(outputs.hidden_states):
    print(f'Layer {i}: maxAbs={h[0,-1,:].float().abs().max().item():.2f}')
"
```

---

## Next Investigation Steps

1. **Verify embedding scaling in DOPPLER**
   - Check if embeddings are scaled by √hidden_size
   - Compare first layer input between DOPPLER and HuggingFace

2. **Add per-operation debug logging**
   - Log attention Q/K/V magnitudes
   - Log attention scores (before and after softmax)
   - Log FFN intermediate values

3. **Check matmul output scaling**
   - Q4K dequant may have incorrect scale factors
   - Compare small matmul outputs between DOPPLER and reference

---

## Files Modified This Session

| File | Change |
|------|--------|
| `docs/DOPPLER-TROUBLESHOOTING.md` | Added note about safetensors/GGUF source formats |
| `models/gemma-3-1b-it-q4/` | Renamed from gemma3-1b-q4 |
| Deleted: `models/gemma3-1b-q4-old/` | Removed duplicate model |

---

## Related Documentation

- [POSITIVE-BIAS-HIDDEN-STATES-POSTMORTEM.md](POSITIVE-BIAS-HIDDEN-STATES-POSTMORTEM.md) - Previous investigation
- [DOPPLER-TROUBLESHOOTING.md](../DOPPLER-TROUBLESHOOTING.md) - Debug guide

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--kernel-plan`, `--kernel-profile`), and the OPFS purge helper.

