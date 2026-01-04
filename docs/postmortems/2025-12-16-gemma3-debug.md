# Gemma 3 1B Inference Debug Post-Mortem

This document covers the debugging session for Gemma 3 1B WebGPU inference, which produced garbage tokens (`<unused16>`) instead of coherent text.

## Symptoms

- Model output: `<unused16>` and other special tokens instead of meaningful text
- Hidden state statistics showed positive bias accumulation through layers:
  - **Our implementation**: L25 mean=686.7, min=71.8, max=1967
  - **HuggingFace reference**: L25 mean=16.8, min=-768, max=46336
- All hidden state values were positive (min=71.8) when they should include negatives (min=-768)
- Token prediction confidence was suspiciously high (86% for wrong token)

## Root Cause: Q4_K Quantization Format Mismatch

The primary bug was in `tools/quantizer.js`. Our quantizer produced data in a **different format** than what the GPU dequantizer (`gpu/kernels/dequant_shared.wgsl`) expected.

### The Mismatch

| Aspect | Our Quantizer (broken) | GPU Kernel (llama.cpp format) |
|--------|------------------------|-------------------------------|
| **Dequant formula** | `q * scale + min` | `d * sc * q - dmin * min` |
| **Min encoding** | Actual min with sign bit in bit 6 | Positive offset to subtract |
| **Byte layout** | Custom interleaved 3-byte pairs | llama.cpp split: bytes 0-3 scales, 4-7 mins, 8-11 packed |

### Why This Caused Positive Bias

The llama.cpp format stores `-actual_min` as a positive value to be subtracted:
```
value = scale * q - min_offset
```

Where `min_offset = -actual_minimum`. For a range like [-0.5, 0.5]:
- `min_offset = 0.5` (positive)
- When q=0: `value = 0 - 0.5 = -0.5` (correct negative)

Our broken quantizer stored the actual minimum with a sign bit, but the GPU kernel ignored the sign bit and always subtracted, causing:
- Negative weights to dequantize as positive
- Loss of all negative values in weight matrices
- Positive bias accumulating through 26 layers

### The Fix

Rewrote `quantizeQ4KBlock()` in `tools/quantizer.js`:
1. Store `-min` as positive offset (the value to subtract)
2. Use llama.cpp byte layout for scales/mins
3. Match the nibble packing format for 4-bit values

## Other Issues Investigated

### 1. Gemma 3 RMSNorm Weight Offset (+1)

**Issue**: Gemma 3 uses `(1 + weight) * normalized` instead of `weight * normalized`.

**Status**: Already fixed in `loader/doppler-loader.js`. The loader applies +1 offset to norm weights during loading via `tryLoadNorm()`.

**Verification**: Checked that norm weights showed correct mean (~5.5 after +1 offset applied to ~4.5 base).

### 2. BF16 to F32 Conversion

**Issue**: Potential sign handling in JavaScript bitwise operations.

**Status**: Verified correct. The code:
```javascript
u32View[0] = bf16[i] << 16;
f32[i] = f32View[0];
```
JavaScript's signed 32-bit wrap-around produces correct bit patterns when stored to Uint32Array.

### 3. Vocab Size Mismatch

**Issue**: LM head matmul dimensions didn't match vocab size.

**Status**: Fixed earlier in the session.

### 4. Sandwich Norm Architecture

**Issue**: Gemma 3 uses a different residual pattern with pre/post feedforward norms.

**Status**: Verified correct in `_processLayerGPU()`. The flow matches HuggingFace:
```
attn_out = attention(input_layernorm(x))
attn_out = post_attention_layernorm(attn_out)
x = x + attn_out
ffn_in = pre_feedforward_layernorm(x)
ffn_out = mlp(ffn_in)
ffn_out = post_feedforward_layernorm(ffn_out)
x = x + ffn_out
```

### 5. QK-Norm Application

**Issue**: Per-head RMSNorm on Q and K projections.

**Status**: Verified correct with `batchSize: numTokens * numHeads` for Q and `batchSize: numTokens * numKVHeads` for K.

### 6. RoPE Configuration

**Issue**: Gemma 3 uses theta=1000000 and specific position handling.

**Status**: Verified correct. `startPos` passed correctly for decode steps.

## Files Modified

| File | Change |
|------|--------|
| `tools/quantizer.js` | Rewrote Q4_K encoding to match llama.cpp format |
| `models/gemma3-1b-q4/` | Re-converted with fixed quantizer |

## Files Created During Debug

| File | Purpose | Status |
|------|---------|--------|
| `inference/pipeline/index.js` | Module re-exports | Kept |
| `inference/pipeline/sampling.js` | Token sampling | Kept |
| `inference/pipeline/config.js` | Model config parsing | Kept |
| `inference/pipeline/embed.js` | Embedding operations | Kept |

## Debugging Techniques Used

1. **Buffer statistics logging**: Added `_debugCheckBuffer()` to track min/max/mean through layers
2. **Weight verification**: Created test scripts to verify weight loading and statistics
3. **Layer-by-layer comparison**: Compared hidden states at L0, L13, L25 against HuggingFace
4. **Quantization round-trip test**: Verified quantize/dequantize preserves negative values

## Key Learnings

1. **Format compatibility matters**: When implementing quantization, match the exact format expected by the dequantizer - not just "close enough"

2. **Sign handling is subtle**: The difference between `+ min` and `- min_offset` seems small but completely breaks inference

3. **Positive bias is a red flag**: When all values become positive through layers, suspect something is clipping or mishandling negative values

4. **Test quantization separately**: A simple round-trip test (quantize -> dequantize -> compare) would have caught this immediately

5. **Check the llama.cpp source**: For Q4_K and other GGML formats, the authoritative reference is llama.cpp's `ggml-quants.c`

## Verification Checklist

After fixing, verify:
- [ ] Quantizer test passes with negative value preservation
- [ ] Hidden state min values are negative at layer outputs
- [ ] Model generates coherent text (not `<unused16>`)
- [ ] Token probabilities are reasonable (not 86% for first token)

## Related Files Reference

- `gpu/kernels/dequant_shared.wgsl`: GPU Q4_K dequantization (llama.cpp format)
- `gpu/kernels/dequant_subgroup.wgsl`: Subgroup-optimized variant
- `tools/quantizer.js`: CPU quantization for model conversion
- `loader/doppler-loader.js`: Weight loading with Gemma 3 norm offset
- `inference/pipeline.js`: Main inference with layer processing

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

