---
name: doppler-debug
description: Debug DOPPLER WebGPU inference issues. Use when investigating model output problems, kernel bugs, hidden state explosions, or decode failures in the browser-based LLM inference engine.
---

# DOPPLER Debug Skill

You are debugging DOPPLER, a browser-native WebGPU LLM inference engine.

## CRITICAL: Systematic Debugging Workflow

When debugging inference issues, follow this systematic workflow:

### Step 1: Run Kernel Correctness Tests FIRST

```bash
# From doppler/reploid directory (browser opens by default)
npm run doppler -- test correctness
```

If any kernel fails, **FIX IT FIRST** - inference bugs are almost always caused by broken kernels.

**Expected results:** All tests PASS except scatter-add (pre-existing issue)

### Step 2: Verify End-to-End Pipeline

```bash
# Quick inference test with debug output
npm run doppler -- bench inference --prompt xs --debug

# Watch specific debug patterns
npm run doppler -- bench inference --prompt xs --debug 2>&1 | grep -E "LAYER|FINAL|logits|Generated"
```

### Step 3: Compare Against Reference Implementation

Before assuming DOPPLER is broken, verify the model works in a reference implementation:

```bash
# Using HuggingFace transformers (Python)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('google/gemma-3-1b-it')
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')
inputs = tokenizer('The color of the sky is', return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0]))
"

# Or using llama.cpp
./llama-cli -m gemma-3-1b-q4_K_M.gguf -p "The color of the sky is" -n 10
```

If reference produces correct output but DOPPLER doesn't, the bug is in DOPPLER.

---

## Model Verification Checklist

### 1. Manifest Configuration

Check `manifest.json` in the converted RDRR model:

```bash
# Key fields to verify:
cat model/manifest.json | jq '{
  vocab_size,
  hidden_size,
  num_layers,
  num_attention_heads,
  num_kv_heads,
  head_dim,
  intermediate_size,
  rms_norm_eps,
  rope_theta,
  rope_local_base_freq,
  sliding_window_pattern,
  scale_embeddings,
  rms_norm_weight_offset
}'
```

**Critical Gemma 3 settings:**
- `scale_embeddings: true` - Embeddings scaled by sqrt(hidden_size)
- `rms_norm_weight_offset: true` - Norm uses (1 + weight) formula
- `rope_theta: 1000000` - Global attention RoPE base
- `rope_local_base_freq: 10000` - Local/sliding attention RoPE base
- `sliding_window_pattern: 6` - Every 6th layer is global

### 2. Weight Verification

```bash
# Check weight statistics during loading
npm run doppler -- bench inference --prompt xs --debug 2>&1 | grep -E "DopplerLoader.*weight|norm.*min|norm.*max"
```

**Expected norm weight ranges for Gemma 3:**
- `input_layernorm`: min ~2.5, max ~55 (before +1 offset)
- `post_attention_layernorm`: min ~-1, max ~28
- `q_norm, k_norm`: min ~-0.75, max ~1.2 (NO +1 offset - standard RMSNorm)

### 3. Tokenizer Verification

```bash
# Test tokenizer produces expected IDs
# In browser console or test:
const tokens = await tokenizer.encode("The color of the sky is");
console.log(tokens);  // Should match HuggingFace tokenizer output
```

### 4. Quantization Verification

For Q4_K models, verify dequantization produces correct values:

```bash
npm run doppler -- test correctness --filter dequant
npm run doppler -- test correctness --filter matmul-q4k
```

---

## Debug Commands Reference

### Kernel Testing

```bash
# All kernel tests (browser opens automatically)
npm run doppler -- test correctness

# Specific kernel
npm run doppler -- test correctness --filter matmul
npm run doppler -- test correctness --filter softmax
npm run doppler -- test correctness --filter rmsnorm

# Q4K quantized matmul tests
npm run doppler -- test correctness --filter q4k
```

### Inference Debugging

```bash
# Quick debug (xs = "The color of the sky is")
npm run doppler -- bench inference --prompt xs --debug

# Layer-by-layer hidden state tracking
npm run doppler -- bench inference --prompt xs --debug 2>&1 | grep -E "LAYER_[0-9]+_LAST"

# Final hidden state and logits
npm run doppler -- bench inference --prompt xs --debug 2>&1 | grep -E "FINAL_HIDDEN|logits|top-5"

# Attention/KV cache debugging
npm run doppler -- bench inference --prompt xs --debug 2>&1 | grep -E "KV|ATTN|attention"

# FFN debugging
npm run doppler -- bench inference --prompt xs --debug 2>&1 | grep -E "FFN|SiLU|gate"

# All output (verbose)
npm run doppler -- bench inference --prompt xs --debug --verbose 2>&1 | head -200
```

### Headless Mode (CI)

```bash
# For CI pipelines where no display is available
npm run doppler -- test correctness --headless
npm run doppler -- bench inference --prompt xs --headless
```

---

## Common Failure Patterns

| Symptom | Likely Cause | Debug Command |
|---------|--------------|---------------|
| Garbage Unicode (Arabic/Russian) | matmul transposeB wrong, Q4K dequant | `--filter matmul-q4k` |
| English but wrong words | Norm weight offset, attention bug | Check manifest `rms_norm_weight_offset` |
| maxAbs > 500 at layer 10+ | Hidden state explosion | `grep "LAYER.*maxAbs"` |
| Zero logits | KV cache not populated | `grep "KV\|hasGPUCache"` |
| NaN/Inf values | Scale overflow | Check dequant d/dmin values |
| First token OK, rest garbage | Decode position bug | Check `startPos` in RoPE |

---

## Hidden State Health Checks

**Healthy value ranges:**

| Stage | Expected maxAbs | Warning Threshold |
|-------|-----------------|-------------------|
| After embedding | <50 | >100 |
| Layer 0-5 | <100 | >200 |
| Layer 10-15 | <200 | >400 |
| Layer 20-25 | <300 | >600 |
| Final hidden | <100 | >200 |
| Logits | <30 | >50 |

**Check with:**
```bash
npm run doppler -- bench inference --prompt xs --debug 2>&1 | \
  grep "LAYER.*maxAbs" | \
  awk -F'maxAbs=' '{print $2}' | \
  awk -F',' '{if ($1 > 500) print "WARNING: Explosion detected: " $1}'
```

---

## Gemma 3 Specific Issues

### Dual RoPE Frequencies
Gemma 3 uses different RoPE bases for local vs global attention:
- **Local (sliding_window)**: `ropeTheta = 10,000`
- **Global (full_attention)**: `ropeTheta = 1,000,000`

Pattern: layers where `i % 6 === 0` are global, others are local.

### Norm Weight Offset
Gemma 3 uses `output = x * (1 + weight) / rms` for layer norms.
**BUT**: `q_norm` and `k_norm` use standard RMSNorm (no +1 offset).

### Sandwich Norm Structure
Each layer has 4 norms:
1. `input_layernorm` (before attention)
2. `post_attention_layernorm` (after attention residual)
3. `pre_feedforward_layernorm` (before FFN)
4. `post_feedforward_layernorm` (after FFN residual)

---

## Key Files to Instrument

| File | Debug Focus |
|------|-------------|
| `inference/pipeline.ts` | Overall flow, token loop |
| `inference/pipeline/layer.ts` | Per-layer processing |
| `inference/pipeline/attention.ts` | KV cache, RoPE, attention |
| `inference/pipeline/ffn.ts` | FFN gate/up/down projections |
| `inference/pipeline/logits.ts` | Final projection, sampling |
| `gpu/kernels/matmul.ts` | Q4K selection, dispatch |
| `loader/doppler-loader.ts` | Weight loading, norm offset |

---

## Build and Test Cycle

```bash
# 1. Make code changes
vim doppler/gpu/kernels/matmul.ts

# 2. Rebuild (required after any .ts changes)
npm run build:doppler

# 3. Test
npm run doppler -- test correctness --filter matmul
npm run doppler -- bench inference --prompt xs --debug
```

**IMPORTANT:** The browser loads JavaScript from `/doppler/dist/`, not TypeScript directly. Changes to `.ts` files won't take effect until rebuilt.

---

## Resources

1. **Troubleshooting Guide**: `doppler/docs/DOPPLER-TROUBLESHOOTING.md`
2. **Postmortems**: `doppler/docs/postmortems/`
   - SOFTMAX-UNIFORM-BUFFER-POSTMORTEM.md - Uniform buffer layout swapped
   - POSITIVE-BIAS-HIDDEN-STATES-POSTMORTEM.md - Hidden state debugging
   - GEMMA3-DEBUG-POSTMORTEM.md - Q4_K quantization issues
3. **Architecture**: `doppler/docs/ARCHITECTURE.md`
4. **TODO**: `doppler/docs/TODO.md` - Current known issues

---

## Related Skills

- **doppler-benchmark**: Run performance benchmarks
- **model-convert**: Convert GGUF/SafeTensors to RDRR format

---

## Current Bug: Gemma 3 "postwar" Output (2025-12-18)

### Status: UNSOLVED

**Symptom**: Model outputs "postwarabisavits Stevens..." instead of "blue" for prompt "The color of the sky is"

### Key Discovery: Hidden States are 10-30x SMALLER than HuggingFace

| Layer | HuggingFace | DOPPLER | Issue |
|-------|-------------|---------|-------|
| After Layer 0 | 227 | 24.47 | 9x too small |
| After Layer 16 | 8,384 | 316.91 | 26x too small |
| Before final norm | 15,168 | 821 | 18x too small |

**The issue is UNDER-accumulation through residual connections.**

### Already Verified CORRECT

1. **BF16 norm weights** - Match exactly after conversion
2. **+1 norm weight offset** - Applied once (not duplicated)
3. **Residual add kernel** - Does proper `a + b`
4. **GELU activation** - Uses correct tanh approximation (0.044715)
5. **Model naming** - Standardized to `gemma-3-1b-it-q4`

### NOT YET TRIED

1. **Embedding scaling verification** - Is ×√hidden_size applied correctly?
2. **Attention Q/K/V magnitude comparison** - Layer-by-layer vs HuggingFace
3. **Q4K dequantization during matmul** - Actual output values
4. **FFN intermediate values** - Gate, up, activation magnitudes

### Likely Root Causes

1. **Embedding scaling** (40%) - May not be scaling by √1152
2. **Attention magnitude** (30%) - Q4K matmul producing too-small outputs
3. **FFN magnitude** (20%) - Gate/up projections under-scaled
4. **RMSNorm bug** (10%) - Dividing by wrong value

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

### Postmortems

- `docs/postmortems/HIDDEN-STATE-UNDERACCUMULATION-2025-12-18.md` - Latest
- `docs/postmortems/POSITIVE-BIAS-HIDDEN-STATES-POSTMORTEM.md` - Previous (disproved)

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

