---
name: doppler-debug
description: Debug DOPPLER WebGPU inference issues. Use when investigating model output problems, kernel bugs, hidden state explosions, or decode failures in the browser-based LLM inference engine.
---

# DOPPLER Debug Skill

You are debugging DOPPLER, a browser-native WebGPU LLM inference engine.

---

## ⛔ CRITICAL: USE THE RIGHT COMMAND ⛔

**When debugging inference issues, you MUST use:**
```bash
npm run debug
```

**NEVER use `npm test -- inference` for debugging.** That is for smoke tests only.

| Task | Correct Command | WRONG Command |
|------|-----------------|---------------|
| Debug garbage output | `npm run debug` | ~~`npm test -- inference`~~ |
| Debug with trace | `npm run debug` | ~~`npm test -- inference`~~ |
| Break on anomaly | `npm run debug -- --break` | ~~`npm test -- --break`~~ |

**The `debug` command:**
- Enables kernel trace automatically
- Keeps browser open for inspection
- Shows tensor stats at each pipeline step
- Detects NaN/Inf/explosions

**The `test -- inference` command:**
- Just checks if model loads and generates tokens
- No trace, no debugging info
- Closes immediately after

---

## Step 0: Check Known Issues First

Before debugging, scan the postmortems for known patterns. The current issue may be related, but may not be.

```bash
cat docs/postmortems/INDEX.md | grep -A 30 "Common Patterns"
```

Common failure patterns:
- Garbage Unicode → Q4K layout or matmul transposeB
- "token=1" in single-token decode → buffer pool padding (false positive)
- Silent kernel failures → WebGPU 'auto' layout issues
- Gemma 3 norms → check +1 weight offset

Full history: `docs/postmortems/INDEX.md`

---

## CLI Commands (3 Distinct Modes)

```bash
npm test         # Correctness (kernel unit tests)
npm run bench    # Performance (tok/s benchmarks)
npm run debug    # Debugging (trace + inspection) ← USE THIS FOR DEBUGGING
```

## Systematic Debugging Workflow

### Step 1: Run Kernel Correctness Tests FIRST

```bash
npm test                        # Quick kernel tests
npm test -- --full              # All kernel tests
npm test -- --filter matmul     # Specific kernel
```

If any kernel fails, **FIX IT FIRST** - inference bugs are almost always caused by broken kernels.

### Step 2: Debug with Kernel Trace

```bash
# Debug mode with trace enabled by default
npm run debug

# Break on first anomaly (NaN/Inf/explosion)
npm run debug -- --break

# Trace specific layers
npm run debug -- --trace-layers 0,5,10
```

The kernel trace shows exactly where values explode:
```
⚠️ [TRACE] Value explosion at L1.post_attn_norm: 5.29 → 105.64 (20.0x)
⚠️ [TRACE] Value explosion at L10.post_attn_norm: 0.56 → 410.82 (736.0x)
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
npm run debug 2>&1 | grep -E "weight|norm.*min|norm.*max"
```

**Expected norm weight ranges for Gemma 3:**
- `input_layernorm`: min ~2.5, max ~55 (before +1 offset)
- `post_attention_layernorm`: min ~-1, max ~28
- `q_norm, k_norm`: min ~0.25, max ~2.2 (WITH +1 offset - Gemma3RMSNorm)

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
npm test -- --filter dequant
npm test -- --filter matmul-q4k
```

---

## Debug Commands Reference

### Kernel Testing

```bash
npm test                          # Quick kernel tests
npm test -- --full                # All kernel tests
npm test -- --filter matmul       # Specific kernel
npm test -- --filter q4k          # Q4K quantized matmul tests
```

### Inference Debugging

```bash
# Debug mode with trace enabled by default
npm run debug

# Explicit trace flag (same as default)
npm run debug -- --trace

# Break on first anomaly
npm run debug -- --break

# Watch trace output
npm run debug 2>&1 | grep -E "TRACE|ANOMALY"
```

### Log Levels

Control loader output verbosity:

| Flag | Level | Shows |
|------|-------|-------|
| (default) | `info` | Phase starts/ends, totals |
| `--verbose` | `verbose` | + Per-shard source, per-layer timing |
| `--trace` | `trace` | + Tensor shapes, dequant ops, buffer details |
| `--quiet` | `silent` | Errors only |

**Browser URL:** `?log=verbose` or `?log=trace`

**Debug mode defaults to `trace`** - shows tensor shapes, dequant ops, and value explosions.

### Benchmarking

```bash
npm run bench                     # Full inference benchmark
npm run bench -- --kernels        # Kernel microbenchmarks
npm run bench -- --runs 3         # Multiple runs for statistics
```

### Headless Mode (CI)

```bash
npm test -- --headless
npm run bench -- --headless
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
npm run debug 2>&1 | grep "LAYER.*maxAbs"
```

---

## Gemma 3 Specific Issues

### Dual RoPE Frequencies
Gemma 3 uses different RoPE bases for local vs global attention:
- **Local (sliding_window)**: `ropeTheta = 10,000`
- **Global (full_attention)**: `ropeTheta = 1,000,000`

Pattern: layers where `i % 6 === 0` are global, others are local.

### Norm Weight Offset
Gemma 3 uses `output = x * (1 + weight) / rms` for ALL norms.
This includes `q_norm` and `k_norm` - they also use Gemma3RMSNorm with +1 offset.

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
npm run build

# 3. Test
npm test -- --filter matmul        # Kernel correctness
npm run debug                      # Debug with trace
```

**IMPORTANT:** The browser loads JavaScript from `/doppler/dist/`, not TypeScript directly. Changes to `.ts` files won't take effect until rebuilt.

---

## Resources

- **Troubleshooting Guide**: `docs/DOPPLER-TROUBLESHOOTING.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Postmortems Index**: `docs/postmortems/INDEX.md` - Key takeaways from resolved bugs

---

## Related Skills

- **doppler-benchmark**: Run performance benchmarks
- **model-convert**: Convert GGUF/SafeTensors to RDRR format

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

