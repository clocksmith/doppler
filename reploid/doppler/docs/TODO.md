# DOPPLER Inference Fix TODO

## Status: Hidden State UNDER-ACCUMULATION Identified (NOT Explosion)

**Last Updated:** 2025-12-18
**Model:** Gemma 3 1B Q4_K_M (converted from safetensors)
**Model Path:** `doppler/models/gemma-3-1b-it-q4/`
**Prompt:** "The color of the sky is"
**Expected:** "blue"
**Actual:** "postwarabisavits Stevens..." (garbage)

---

## Summary

**KEY DISCOVERY (2025-12-18):** Hidden states are 10-30x SMALLER than HuggingFace reference, NOT exploding!

| Layer | HuggingFace | DOPPLER | Issue |
|-------|-------------|---------|-------|
| After Layer 0 | 227 | 24.47 | 9x too small |
| After Layer 16 | 8,384 | 316.91 | 26x too small |
| Before final norm | 15,168 | 821 | 18x too small |

The issue is **under-accumulation through residual connections**. Previous hypothesis about "explosion" was wrong - HuggingFace naturally produces maxAbs=15,000+ before final normalization.

### Recent Fix: q_norm/k_norm +1 Offset Bug (doppler-loader.ts:1224-1227)
Gemma 3's per-head Q/K normalizations (q_norm, k_norm) use standard RMSNorm, NOT the (1+weight) formula used by layer norms. These weights should NOT have +1 offset applied.

**Before fix:** `blue` logit = -5.53, output = Arabic garbage
**After fix:** `blue` logit = +1.49, output = "postwar" (English, but still wrong)

### Key Finding: Model Works in Transformers, Not DOPPLER
The model was re-converted from fresh HuggingFace SafeTensors with correct flags:
- Manifest now has `scale_embeddings: true` and `rms_norm_weight_offset: true`
- **HuggingFace transformers produces correct output** ("The color of the sky is a constant, but its appearance changes...")
- **DOPPLER still produces garbage** - the issue is in DOPPLER's implementation, NOT the model conversion

### Gemma 3 Norm Weight Characteristics
Gemma 3 has unusually high norm weights (this is intentional, not corruption):
- `input_layernorm`: min=2.5, max=55.8 (before +1 offset)
- `pre_feedforward_layernorm`: min=-1.0, max=28.1
- `post_feedforward_layernorm`: min=-1.0, max=66.5
- `final_norm`: min=-1.0, max=46.0

After +1 offset (Gemma 3 formula: `output = x * (1 + weight) / rms`), these become 3-67x multipliers.

### Kernel Test Results
| Test | Status | Notes |
|------|--------|-------|
| matmul (F32) | PASS | |
| matmul-q4k (M>1, K=256) | PASS | Tests subgroup column fix |
| matmul-q4k-large (K=1152) | PASS | Tests non-256-aligned K |
| attention | PASS | |
| rmsnorm | PASS | |
| softmax | PASS | Fixed uniform buffer layout (was FAIL with maxError=0.137) |
| rope | PASS | |
| silu | PASS | |
| gather | PASS | |
| dequant | PASS | |
| residual | PASS | |
| topk | PASS | |
| moe-gather | PASS | |
| scatter-add | FAIL | (pre-existing) |

### Inference Debug Output (After q_norm/k_norm Fix)
```
LAYER_0_LAST[pos=15]:  maxAbs=24.47
LAYER_12_LAST[pos=15]: maxAbs=135.20
LAYER_17_LAST[pos=15]: maxAbs=723.79  <- Major explosion here
LAYER_25_LAST[pos=15]: maxAbs=821.41  <- Should be <100
FINAL_HIDDEN[pos=15]: sample=[-287.39, -164.24, 595.77, 31.52, 90.93]  <- Mixed signs (correct)
Logits: min=-26.03, max=22.03
Top-5: "postwar"(12.6%), "uro"(10.0%), "слови"(8.5%)  <- Improved from Arabic garbage
Specific tokens: blue=1.49, BLUE=2.17, Kaw=-5.67  <- blue logit now positive!
```

### Layer-by-Layer Growth Pattern
The hidden state explosion primarily occurs in layers 14-17:
- Layer 0-10: Moderate growth (24 → 56, ~2.3x)
- Layer 10-14: Accelerating (56 → 206, ~3.7x)
- Layer 14-17: Major explosion (206 → 724, ~3.5x in 3 layers)
- Layer 17-25: Stable (724 → 821)

Layers 14-17 are all local attention (sliding_window) layers. The pattern suggests something specific to these layers is amplifying values.

---

## Completed Fixes

### Kernel Fixes
- [x] Fix q4_fused_batched subgroup column mixing (`matmul_q4_fused.wgsl:207-289`)
  - Changed from 4 cols/workgroup to 1 col/workgroup (64 threads/col)
  - Prevents subgroup mixing when sg_size=32
- [x] Fix wg_sums array overflow (`matmul_q4_fused.wgsl:47`)
  - Changed from array<f32, 8> to array<f32, 32> (supports sg_size >= 8)
- [x] Update dispatch logic (`matmul.ts:362-367, 598-603`)
  - Changed workgroupsX = N (was N/4) for new kernel layout
- [x] Add subgroup capability check (`matmul.ts:215-220`)
  - Error thrown if subgroups not supported

### Pipeline Fixes
- [x] Change transposeB: true → 'auto' in all pipeline files
- [x] Fix attention decode array sizing (256 elements)
- [x] Fix recordGather dtype detection
- [x] Fix softmax uniform buffer layout (`softmax.ts:45-48, 177-180`)
  - innerSize/outerSize were swapped in uniform struct layout
  - WGSL expects innerSize at offset 0, outerSize at offset 4
  - TypeScript was writing them in wrong order

### Loader Fixes
- [x] Fix q_norm/k_norm +1 offset bug (`doppler-loader.ts:1224-1227`)
  - Changed `tryLoadNorm` to `tryLoad` for q_norm and k_norm weights
  - These per-head normalizations use standard RMSNorm, NOT Gemma 3's (1+weight) formula
  - Raw weight range: min=-0.75, max=1.20 (should NOT be shifted by +1)
  - Impact: "blue" logit improved from -5.53 to +1.49

### Test Infrastructure
- [x] Added `matmul-q4k` test to CLI (M>1, K=256)
- [x] Added `matmul-q4k-large` test to CLI (K=1152)
- [x] Added Q4K logging to matmul.ts
- [x] Added Q4K loading logs to doppler-loader.ts

---

## Known Issues (Partially Addressed)

### Hidden State Under-Accumulation (Primary Remaining Issue)

**UPDATE 2025-12-18:** This is NOT an explosion - hidden states are actually 10-30x SMALLER than expected!

Comparison with HuggingFace transformers reference:
- HuggingFace Layer 0: maxAbs=227, DOPPLER: maxAbs=24.47 (9x smaller)
- HuggingFace Layer 16: maxAbs=8,384, DOPPLER: maxAbs=316.91 (26x smaller)
- HuggingFace before final norm: maxAbs=15,168, DOPPLER: maxAbs=821 (18x smaller)

The residual stream is not accumulating properly. Divergence starts at Layer 0 output (9x too small) and compounds through layers.

### What's Been Verified CORRECT (2025-12-18):
- BF16 norm weights match exactly after conversion
- +1 norm weight offset applied once (not duplicated)
- Residual add kernel does proper `a + b`
- GELU activation uses correct 0.044715 coefficient
- Model naming standardized to `gemma-3-1b-it-q4`

### What's Been Verified Working:
- Q4K dequantization formula is correct (tests pass)
- RMSNorm formula: `output = x * weight / rms` (tests pass)
- GELU formula: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))` (matches gelu_pytorch_tanh)
- RoPE frequencies: local=10K, global=1M (correctly configured)
- Layer types: correctly alternates sliding_attention/full_attention based on sliding_window_pattern=6
- Weight shapes: all match Gemma 3 architecture (q_proj=[1024,1152], o_proj=[1152,1024], etc.)
- Sandwich norm structure: 4 norms per layer correctly applied

---

## Remaining Investigation Paths (Priority Order)

### 1. Embedding Scaling Verification (HIGHEST PRIORITY - 40%)
Gemma 3 scales embeddings by √hidden_size. Verify this is happening:
```python
# HuggingFace
embeds = model.model.embed_tokens(input_ids)  # maxAbs ~19
scaled = embeds * (1152 ** 0.5)  # maxAbs ~653
```
Check if DOPPLER's embedding output matches the scaled value, not raw.

### 2. Attention Output Magnitude (30%)
Compare Q, K, V, and attention output magnitudes layer-by-layer:
- Add debug logging to `attention.ts` for Q/K/V projections
- Compare `attnOutput` before and after RMSNorm
- Verify Q4K dequantization produces correct magnitude outputs

### 3. FFN Intermediate Values (20%)
Compare gate, up, activation, down outputs:
- Add debug to `layer.ts` for `gateUpOutput`, `activated`, `downOutput`
- Verify GELU activation doesn't clip values

### 4. RMSNorm Scaling Bug (10%)
RMSNorm may be dividing by wrong RMS value:
- Add debug to `runRMSNorm` to log input RMS vs output magnitude
- Compare with HuggingFace RMSNorm behavior

### Reference Comparison Script
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('google/gemma-3-1b-it', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')
inputs = tokenizer('The color of the sky is', return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
for i, h in enumerate(outputs.hidden_states):
    print(f'Layer {i}: maxAbs={h[0,-1,:].float().abs().max().item():.2f}')
# Expected: Layer 0=227, Layer 16=8384, Final=60
```

---

## Debug Commands

```bash
# Run inference with debug output
doppler bench inference --prompt xs --debug 2>&1 | grep -E "LAYER|FINAL|logits|Generated"

# Run kernel tests
doppler test correctness

# Run specific Q4K test
doppler test correctness --filter q4k

# Full verbose output
doppler bench inference --prompt xs --debug --verbose 2>&1 | head -100
```

---

## Related Documentation

- [HIDDEN-STATE-UNDERACCUMULATION-2025-12-18.md](postmortems/HIDDEN-STATE-UNDERACCUMULATION-2025-12-18.md) - LATEST (key discovery)
- [POSITIVE-BIAS-HIDDEN-STATES-POSTMORTEM.md](postmortems/POSITIVE-BIAS-HIDDEN-STATES-POSTMORTEM.md) - Previous (disproved)
- [SOFTMAX-UNIFORM-BUFFER-POSTMORTEM.md](postmortems/SOFTMAX-UNIFORM-BUFFER-POSTMORTEM.md) - RESOLVED
- [DOPPLER-TROUBLESHOOTING.md](DOPPLER-TROUBLESHOOTING.md) - Debug guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - System overview

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

