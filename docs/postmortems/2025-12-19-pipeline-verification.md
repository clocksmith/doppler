# Gemma 3 1B Pipeline Verification Post-Mortem

This document covers the systematic pipeline verification process for Gemma 3 1B WebGPU inference, which identified an FFN explosion issue causing garbage token output.

## Symptoms

- Model output: garbage tokens (Telugu, Kannada, Japanese scripts, `<unused>` tokens)
- Near-uniform logit distribution (~3% top token confidence)
- Top-5 tokens: "ನ"(3.2%), "<unused16>"(1.6%), "మా"(1.3%), "マ"(1.3%), "<end_of_turn>"(0.8%)
- FFN down projection explosion at Layer 0: min=-3078, max=756, mean=-268

## Root Cause: FFN Value Explosion

The primary issue was extreme value magnitudes in the FFN down projection output, which propagated through layers and corrupted the final logits.

### The Explosion

| Stage | Min | Max | Mean | Status |
|-------|-----|-----|------|--------|
| Embedding (scaled) | -1.28 | 25.19 | 0.05 | OK |
| Attention output | -49.4 | 59.2 | 7.38 | Elevated |
| FFN down proj | -3078 | 756 | -268 | Exploded |
| Post-FFN norm | normalized | normalized | normalized | Masked issue |
| Final logits | -19.57 | 20.06 | - | Near-uniform |

### Why This Caused Garbage Output

The post-FFN sandwich norm masked the explosion by normalizing values, but the information was already corrupted. This led to:
- Near-uniform logit distribution
- Model unable to distinguish meaningful tokens
- Non-Latin scripts and unused tokens receiving similar probabilities

### Resolution

This debugging session identified the FFN explosion, which was later traced to the Q4_K quantization format mismatch documented in `2025-12-16-gemma3-debug.md`. The quantizer was producing weights with incorrect sign handling, causing negative weights to dequantize as positive values.

## Pipeline Steps Verified

### Verified Correct

| Step | Component | Verification |
|------|-----------|--------------|
| 1 | Tokenization | 16 tokens for "the color of the sky is" |
| 2 | Embedding lookup | Buffer size 2GB, reasonable values |
| 3 | Gather kernel | min=-0.0376, max=0.0618, mean=-0.0001 |
| 4 | Embedding scaling | sqrt(1152)=33.94, verified multiply |
| 5 | Q4_K dequant (q_proj) | 0/256 mismatches, max error 0.000014 |
| 6 | Q4_K dequant (gate_proj) | 0/256 mismatches, max error 0.000019 |

### Pending at Time of Debug

- Layer 0 RMSNorm weight values
- QKV projections
- QK-Norm (Gemma-specific)
- RoPE positional encoding
- Attention computation (values seemed large)
- Sandwich norm patterns
- Layers 1-25
- Final layer norm
- LM head projection

## Debugging Techniques Used

1. **Step-by-step pipeline logging**: Verified each stage from tokenization to logits
2. **Buffer statistics**: Tracked min/max/mean at each pipeline stage
3. **GPU vs CPU comparison**: Verified dequantization against CPU reference
4. **Value explosion detection**: Identified FFN down projection as explosion point
5. **Architecture verification**: Confirmed Gemma 3 sandwich norm pattern

## Key Learnings

1. **Sandwich norms can mask issues**: The post-FFN norm normalized exploded values, making the problem less obvious until final logits

2. **Track values through full pipeline**: The explosion was only visible by checking FFN output before normalization

3. **Near-uniform logits indicate corruption**: When top token confidence is ~3%, the model has lost meaningful signal

4. **Gemma 3 has unique architecture**: Sandwich norms (pre/post for both attention and FFN) require careful verification

5. **Quantization verification isn't enough**: Dequantization tests passed, but the underlying quantizer was still broken

## Verification Checklist

After fixing, verify:
- [ ] FFN down projection values are bounded (not >1000)
- [ ] Logit distribution is peaked (top token >10%)
- [ ] Output tokens are Latin script for English prompts
- [ ] Layer-by-layer hidden states match HuggingFace reference

## Related Files

- `inference/pipeline.js`: Main inference with layer processing
- `gpu/kernels/matmul*.wgsl`: Matrix multiplication kernels
- `gpu/kernels/rmsnorm.wgsl`: RMSNorm implementation
- `loader/doppler-loader.js`: Weight loading with Gemma 3 norm offset
- `2025-12-16-gemma3-debug.md`: Root cause analysis

## See Also

- `2025-12-16-gemma3-debug.md`: Documents the Q4_K quantization fix that resolved this issue

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `../style/WGSL_STYLE_GUIDE.md` for runtime kernel modes and the OPFS purge helper.

