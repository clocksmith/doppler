# DOPPLER Debug Guide

Comprehensive debugging strategies for DOPPLER WebGPU inference issues. Written for future developers and Claude agents.

**Note:** All CLI commands now run headed (visible browser) by default. Use `--headless` for CI.

---

## Quick Start: Systematic Debug Workflow

### 1. Run Kernel Tests First
```bash
npm run doppler -- test correctness
```
If any kernel fails, **fix it first**. Expected: all PASS except scatter-add.

### 2. Run Inference Debug
```bash
npm run doppler -- bench inference --prompt xs --debug
```

### 3. Compare Against Reference
```bash
# HuggingFace transformers (ground truth)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('google/gemma-3-1b-it')
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')
inputs = tokenizer('The color of the sky is', return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0]))
"
```

If reference works but DOPPLER doesn't, the bug is in DOPPLER implementation.

---

## End-to-End Model Verification

### Manifest Configuration Checklist

Check the converted RDRR model's `manifest.json`.

**Source formats:** DOPPLER's `convert-cli` supports both **safetensors** (HuggingFace) and **GGUF** (llama.cpp) as input formats. The resulting RDRR format is the same regardless of source. If debugging issues, compare against the original source (HuggingFace for safetensors, llama.cpp for GGUF).

```bash
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
  rms_norm_weight_offset,
  activation
}'
```

**Critical Gemma 3 settings:**
| Field | Expected Value | Purpose |
|-------|----------------|---------|
| `scale_embeddings` | `true` | Scale by sqrt(hidden_size) |
| `rms_norm_weight_offset` | `true` | Use (1 + weight) formula |
| `rope_theta` | `1000000` | Global attention RoPE base |
| `rope_local_base_freq` | `10000` | Local attention RoPE base |
| `sliding_window_pattern` | `6` | Every 6th layer is global |
| `activation` | `gelu_pytorch_tanh` | Gemma 3 uses GELU, not SiLU |

### Weight Statistics Verification

```bash
# During inference debug, check weight loading
npm run doppler -- bench inference --prompt xs --debug 2>&1 | grep -E "weight|norm.*min|norm.*max"
```

**Expected Gemma 3 norm weight ranges:**
- `input_layernorm`: min ~2.5, max ~55 (before +1 offset)
- `post_attention_layernorm`: min ~-1, max ~28
- `q_norm, k_norm`: min ~-0.75, max ~1.2 (NO +1 offset!)

### Tokenizer Verification

```javascript
// In browser console:
const tokens = await tokenizer.encode("The color of the sky is");
console.log("DOPPLER tokens:", tokens);

// Compare with HuggingFace:
// from transformers import AutoTokenizer
// t = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')
// print(t.encode("The color of the sky is"))
```

Token IDs must match exactly.

### Quantization Verification (Q4_K)

```bash
# Verify dequantization produces correct values
npm run doppler -- test correctness --filter dequant
npm run doppler -- test correctness --filter matmul-q4k
npm run doppler -- test correctness --filter matmul-q4k-large
```

---

## Quick Diagnosis Table

| Symptom | Likely Cause | First Check |
|---------|--------------|-------------|
| Garbage tokens (`<unused>`, non-Latin scripts) | Quantization format mismatch | Q4_K dequant round-trip test |
| Positive bias through layers | Missing negative values | Weight min/max statistics |
| **Last position ALL POSITIVE** | Q4_K dequant or attention bug | Position-specific hidden state debug |
| FFN explosion (values >1000) | SiLU gating bug or weight corruption | FFN down projection stats |
| Near-uniform logits (~3% top) | Information destroyed early | Layer-by-layer hidden state tracking |
| Zero embeddings for high token IDs | 2D dispatch linearization bug | Test token ID 8192+ vs 0-100 |
| Kernel "runs" but outputs zeros | Bind group layout mismatch | Use explicit layout, not 'auto' |
| Decode broken, prefill works | KV cache or position indexing | Check `startPos`, `kvLen` values |
| Debug readbacks show zeros | CommandRecorder batching | Add `!recorder` check before readback |

---

## 1. Tensor Shape Verification

Always verify buffer sizes match expected dimensions:

```typescript
// Add to any kernel call
console.log(`Q size: ${Q.size}, expected: ${numTokens * numHeads * headDim * 4}`);
console.log(`K size: ${K.size}, expected: ${numTokens * numKVHeads * headDim * 4}`);
```

### Matmul Dimension Checklist

Matmul computes `A[M,K] @ B[K,N] -> C[M,N]`. Verify:
- Input A: `[numTokens, inputDim]`
- Weight B: `[inputDim, outputDim]` (or transposed)
- Output C: `[numTokens, outputDim]`

Wrong `transposeB` flag causes silent corruption.

---

## 2. Toggleable Debug Categories

DOPPLER has a toggleable debug system with category-based filtering. This allows surgical debugging without noise.

### Browser Console API

```javascript
// In browser DevTools console:
DOPPLER.debug()                              // Show current config & help
DOPPLER.debug({ embed: true, logits: true }) // Enable specific categories
DOPPLER.debug('quick')                       // Use preset (embed + logits + sample)
DOPPLER.debug('layers')                      // Layer entry/exit tracing
DOPPLER.debug('attention')                   // Attention + KV cache
DOPPLER.debug('full')                        // Everything (verbose!)
DOPPLER.debug('off')                         // Disable all

// Layer-specific debugging
DOPPLER.debugLayer(0)                        // Debug only layer 0
DOPPLER.debugLayer([0, 1, 25])               // Debug specific layers

// Buffer stats (expensive - requires GPU readback)
DOPPLER.debugBuffers(true)                   // Enable buffer inspection
```

### Debug Categories

| Category | What it logs |
|----------|--------------|
| `embed` | Embedding output (tokens, maxAbs, sample values) |
| `layer` | Layer entry/exit, hidden state stats |
| `attn` | Attention computation (Q/K/V, kvLen, startPos) |
| `ffn` | FFN gate/up/down stats |
| `kv` | KV cache read/write/init/clear |
| `logits` | Logits computation, top-k tokens |
| `sample` | Sampling decisions (token, prob, temp) |
| `io` | GPU buffer read/write operations |
| `perf` | Performance timing |
| `all` | Enable everything |

### Log Levels (Unified System)

DOPPLER uses a unified logging system controlled by CLI flags or browser URL params:

| CLI Flag | URL Param | Level | Shows |
|----------|-----------|-------|-------|
| (default) | `?log=info` | info | Phase starts/ends, totals |
| `--verbose` | `?log=verbose` | verbose | + Per-shard source, per-layer timing |
| `--trace` | `?log=trace` | trace | + Tensor shapes, dequant ops, buffer details |
| `--quiet` | `?log=silent` | silent | Errors only |

**Defaults by mode:**
- `test`, `bench`: info (clean summaries)
- `debug`: verbose (shows shard sources and layer timing)

**Example output at verbose level:**
```
[Loader] Shards: 0-15 (640.0 MB total)
[Loader] Shard 0: RAM (64.0 MB)
[Loader] Shard 1: OPFS (64.0 MB, 0.05s)
[Loader]  Shard 2: network (64.0 MB, 0.31s @ 206.5 MB/s)
...
[Loader] Layers: 0-25
[Loader]   Layer 0: 0.12s
[Loader]   Layer 1: 0.08s
...
[Loader] Complete: 640.0 MB in 2.34s (273.5 MB/s)
```

### CLI Log Forwarding (IMPORTANT)

The DOPPLER CLI (`doppler-cli.ts`) runs Playwright and filters browser console logs before forwarding to stdout. Only logs with these tags are shown:

```
[Benchmark], [Pipeline], [Loader], [GPU], [Kernel], [Layer], [KERNEL], [KV], [ATTN], [FFN], ERROR, WARN
```

**If your logs don't appear:**
1. Check your grep pattern includes the tag (e.g., `Loader` to match loader output)
2. Use `--verbose` to enable verbose-level loader logs
3. Some debug readbacks skip when using CommandRecorder (batched mode) - this is by design

```bash
# Show shard sources and layer timing
doppler bench --verbose 2>&1 | grep -E "Loader.*Shard|Loader.*Layer"

# Show everything including tensor details
doppler debug --trace 2>&1 | head -200

# Quiet mode (errors only)
doppler bench --quiet
```

### OPFS Cache Persistence (Faster Reruns)

The CLI uses a persistent Playwright profile directory to preserve browser storage between runs. This includes the OPFS model cache, so the second run should skip downloads.

- Default profile dirs:
  - Tests: `doppler/.test-cache/`
  - Inference benchmarks: `doppler/.benchmark-cache/`
- Override with `--profile-dir <path>` (relative to `doppler/` or absolute)

```bash
# Reuse the same profile across runs (warm OPFS)
doppler bench inference --prompt xs --profile-dir .benchmark-cache

# Use a fresh profile for a cold-start run
doppler bench inference --prompt xs --profile-dir .benchmark-cache-cold
```

### Log Format for Post-Filtering

All logs use a consistent format: `[CATEGORY][L{layer}][S{step}] message`

This enables grep-based filtering:

```bash
# Filter for specific categories
doppler bench inference --prompt xs 2>&1 | grep -E "^\[LOGITS\]"
doppler bench inference --prompt xs 2>&1 | grep -E "^\[LAYER\]\[L0\]"
doppler bench inference --prompt xs 2>&1 | grep -E "^\[ATTN\]|\[FFN\]"

# Watch layer 0 through decode steps
doppler bench inference --prompt xs 2>&1 | grep "\[L0\]" | head -20
```

### Debug Options

```javascript
DOPPLER.debug(
  { layer: true, attn: true },  // Categories to enable
  {
    layers: [0, 1],             // Only log these layers
    maxDecodeSteps: 5,          // Only log first N decode steps
    maxAbsThreshold: 10000,     // Warn on value explosion
    bufferStats: true,          // Enable GPU buffer readback (expensive)
  }
);
```

### Presets

| Preset | Categories | Use Case |
|--------|------------|----------|
| `quick` | embed, logits, sample | Quick sanity check |
| `layers` | layer | Watch hidden state flow |
| `attention` | attn, kv | Debug attention issues |
| `full` | all | Comprehensive trace |
| `perf` | perf | Performance timing only |

---

## 3. Pipeline Stage Debugging

### Add Strategic Logging

```typescript
// In layer.ts, before/after each stage
async function debugCheckBuffer(
  buffer: GPUBuffer,
  label: string,
  numTokens: number,
  expectedDim?: number
): Promise<void> {
  const data = await readBufferF32(buffer);
  const min = Math.min(...data);
  const max = Math.max(...data);
  const mean = data.reduce((a, b) => a + b, 0) / data.length;
  const maxAbs = Math.max(Math.abs(min), Math.abs(max));

  console.log(`[${label}] min=${min.toFixed(2)}, max=${max.toFixed(2)}, mean=${mean.toFixed(2)}, maxAbs=${maxAbs.toFixed(2)}`);

  // Red flags
  if (min >= 0) console.warn(`WARNING: All values positive - check sign handling`);
  if (maxAbs > 1000) console.warn(`WARNING: Value explosion detected`);
}
```

### Expected Value Ranges

| Stage | Healthy min | Healthy max | Healthy maxAbs |
|-------|-------------|-------------|----------------|
| Embedding (scaled) | -2 | 30 | <50 |
| Post-attention | -100 | 100 | <150 |
| FFN down proj | -500 | 500 | <1000 |
| Final hidden | -50 | 50 | <100 |
| Logits | -20 | 20 | <30 |

---

## 3. Position-Specific Debugging (Advanced)

When debugging issues that only affect certain token positions, global buffer stats can hide the problem.

### The Problem

Hidden state statistics averaged across all positions may look fine:
```
HIDDEN_STATES: min=-100, max=200, mean=50  // Looks okay
```

But position-specific stats reveal issues:
```
HIDDEN[pos=0]: [-97, -21, -76, -9, 117]    // Mixed signs - correct
HIDDEN[pos=6]: [183, 42, 201, 63, 294]     // ALL POSITIVE - bug!
```

### Position-Specific Debug Pattern

```typescript
// Read hidden state at SPECIFIC position (e.g., last token for logits)
const targetPos = numTokens - 1;  // Last position
const posOffset = targetPos * hiddenSize * 4;  // Byte offset
const sampleSize = Math.min(128, hiddenSize * 4);

const staging = device.createBuffer({
  size: sampleSize,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
});

const enc = device.createCommandEncoder();
enc.copyBufferToBuffer(hiddenStates, posOffset, staging, 0, sampleSize);
device.queue.submit([enc.finish()]);

await staging.mapAsync(GPUMapMode.READ);
const data = new Float32Array(staging.getMappedRange().slice(0));
staging.unmap();
staging.destroy();

// Check for position-specific issues
const allPositive = Array.from(data).every(x => x > 0);
const allNegative = Array.from(data).every(x => x < 0);
if (allPositive) console.warn(`[pos=${targetPos}] ALL POSITIVE - check sign handling`);
if (allNegative) console.warn(`[pos=${targetPos}] ALL NEGATIVE - unusual`);

console.log(`[pos=${targetPos}] sample: [${data.slice(0, 5).map(x => x.toFixed(2))}]`);
```

### When to Use Position-Specific Debug

- **Garbage token output**: Check last position hidden state (used for logits)
- **Decode works but prefill broken** (or vice versa): Compare pos=0 vs pos=N-1
- **Long sequences fail**: Check positions near context length boundary
- **First token wrong**: Check pos=0 embedding + first layer output

### CommandRecorder Timing Gotcha

**CRITICAL**: When using CommandRecorder (batched mode), buffers aren't populated until submit!

```typescript
// WRONG - will read zeros
const output = await recordMatmul(recorder, A, B, M, N, K);
const data = await readBuffer(output);  // Returns zeros!

// RIGHT - check for recorder before debug readback
if (!recorder) {
  const data = await readBuffer(output);  // Works - immediate submit
} else {
  console.log('(skipping debug - batched mode)');
}
```

---

## 4. Common Bug Patterns (Consolidated from Postmortems)

These patterns are consolidated from actual debugging sessions. Each links to its detailed postmortem.

### Pattern A: Uniform Buffer Layout Mismatch
**Postmortem**: [SOFTMAX-UNIFORM-BUFFER-POSTMORTEM.md](postmortems/SOFTMAX-UNIFORM-BUFFER-POSTMORTEM.md)

**Symptom**: Kernel correctness test fails, wrong results despite no errors.

**Root cause**: TypeScript writes uniform fields in different order than WGSL struct expects.

**Quick check**:
```bash
# Compare WGSL struct definition with TypeScript write order
grep -A 10 "struct.*Uniforms" gpu/kernels/softmax.wgsl
grep -A 5 "uniformView.setUint32" gpu/kernels/softmax.ts
```

**Fix**: Add comments documenting WGSL layout at every uniform write:
```typescript
// WGSL struct: { innerSize: u32, outerSize: u32, temperature: f32, _pad: u32 }
uniformView.setUint32(0, innerSize, true);   // offset 0
uniformView.setUint32(4, outerSize, true);   // offset 4
```

### Pattern B: Q4_K Quantization Format Mismatch
**Postmortem**: [GEMMA3-DEBUG-POSTMORTEM.md](postmortems/GEMMA3-DEBUG-POSTMORTEM.md)

**Symptom**: All dequantized values positive, no negative weights.

**Root cause**: Quantizer stores `min` differently than llama.cpp format. Must store `-actual_min` as positive offset.

**Quick check**:
```bash
# Round-trip test
npm run doppler -- test correctness --filter dequant
```

**Fix**: `value = d * sc * q - dmin * min` (subtract, not add)

### Pattern C: 2D Dispatch Without Linearization
**Postmortem**: [BF16-2D-DISPATCH-POSTMORTEM.md](postmortems/BF16-2D-DISPATCH-POSTMORTEM.md)

**Symptom**: Works for small tensors, zeros/garbage for large tensors (>65K workgroups).

**Root cause**: Kernel ignores `global_id.y` in 2D dispatch. WebGPU limits 65535 workgroups per dimension.

**Quick check**:
```bash
# Test high token IDs
npm run doppler -- bench inference --text "Test token 10000" --debug
```

**Fix**:
```wgsl
let linear_idx = global_id.y * (uniforms.workgroupsX * WORKGROUP_SIZE) + global_id.x;
```

### Pattern D: 'auto' Layout Silent Failure
**Postmortem**: [MOE-EXPLICIT-LAYOUT-POSTMORTEM.md](postmortems/MOE-EXPLICIT-LAYOUT-POSTMORTEM.md)

**Symptom**: Kernel runs without errors but outputs all zeros.

**Root cause**: `layout: 'auto'` with multi-entry-point shaders—WebGPU silently ignores binding mismatches.

**Quick check**: Create minimal test kernel with single binding to isolate.

**Fix**: Always use explicit bind group layout for complex shaders:
```typescript
const layout = device.createBindGroupLayout({ entries: [/* ALL bindings */] });
```

### Pattern E: FFN Value Explosion (Masked by Sandwich Norm)
**Postmortem**: [PIPELINE-VERIFICATION-POSTMORTEM.md](postmortems/PIPELINE-VERIFICATION-POSTMORTEM.md)

**Symptom**: Near-uniform logits (<10% top token probability).

**Root cause**: FFN explodes but post-FFN norm masks it. Information already destroyed.

**Quick check**:
```bash
# Check FFN values BEFORE normalization
npm run doppler -- bench inference --prompt xs --debug 2>&1 | grep "FFN.*down\|FFN.*FINAL"
# Values > 1000 = explosion
```

**Fix**: Track values at every stage BEFORE normalization.

### Pattern F: Hidden State Explosion
**Postmortem**: [postmortems/INDEX.md](postmortems/INDEX.md) - See q_norm/k_norm and Q4K sections

**Symptom**: maxAbs grows from ~20 to 800+ through layers. Output is garbage Unicode.

**Root cause**: q_norm/k_norm weights missing +1 offset (Gemma 3 uses `(1 + weight)` formula for ALL norms), combined with Q4K layout mismatch causing fallback to dequantized weights.

**Quick check**:
```bash
doppler debug 2>&1 | grep -E "TRACE|explosion"
```

**Fix**: Use `getNormWeightBuffer()` for q_norm/k_norm in attention.ts. Reconvert model after loader fix.

---

## 4.1 Experimental Debug Techniques

### One-Liner Debug Scripts

```bash
# Watch hidden state explosion in real-time
npm run doppler -- bench inference --prompt xs --debug 2>&1 | \
  grep -E "LAYER_[0-9]+.*maxAbs" | \
  while read line; do
    abs=$(echo "$line" | grep -oP 'maxAbs=[\d.]+' | cut -d= -f2)
    [ $(echo "$abs > 500" | bc -l) -eq 1 ] && echo "EXPLOSION: $line" || echo "$line"
  done

# Compare logit rankings for specific tokens
npm run doppler -- bench inference --prompt xs --debug 2>&1 | \
  grep -E "blue=|BLUE=|sky=" | tail -5

# Extract just the layer-by-layer maxAbs values for plotting
npm run doppler -- bench inference --prompt xs --debug 2>&1 | \
  grep "LAYER.*maxAbs" | \
  sed 's/.*LAYER_\([0-9]*\).*maxAbs=\([0-9.]*\).*/\1 \2/' > /tmp/layer_maxabs.dat
```

### Diff Against Reference Implementation

```bash
# Run same prompt through HuggingFace transformers
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained('google/gemma-3-1b-it', torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')

inputs = tokenizer('The color of the sky is', return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

for i, hidden in enumerate(outputs.hidden_states):
    h = hidden[0, -1, :5].tolist()  # Last token, first 5 values
    print(f"Layer {i}: {[f'{x:.2f}' for x in h]}")
EOF
```

### Isolate Specific Layer

```typescript
// Add to layer.ts for surgical debugging
if (layerIdx === 14) {  // Explosion starts here
  const data = await readBufferF32(hiddenStates);
  console.log(`[DEBUG_L14] Before attention:`, {
    min: Math.min(...data),
    max: Math.max(...data),
    sample: data.slice(0, 10)
  });
}
```

### Binary Search for Bug Location

```bash
# If output is garbage, binary search which layer breaks it
for layer in 0 5 10 15 20 25; do
  echo "Testing up to layer $layer"
  # Modify pipeline to exit early at $layer
  npm run build:doppler
  npm run doppler -- bench inference --prompt xs --debug 2>&1 | grep "top-5"
done
```

### Buffer Content Comparison

```bash
# Dump buffer contents for offline analysis
npm run doppler -- bench inference --prompt xs --debug --verbose 2>&1 | \
  grep -A 20 "FINAL_HIDDEN" > /tmp/doppler_hidden.txt

# Compare with previous run
diff /tmp/doppler_hidden_good.txt /tmp/doppler_hidden.txt
```

---

## 5. Prefill vs Decode Issues

### Decode-Specific Checklist

| Check | Expected Value | How to Verify |
|-------|---------------|---------------|
| `currentSeqLen` | Increments each step | Log in pipeline.generate() |
| `startPos` for RoPE | `currentSeqLen` (not 0) | Log in runRoPE() |
| `kvLen` for attention | `currentSeqLen + numTokens` | Log in runAttention() |
| `startPosForMask` | `currentSeqLen` | Log in attention.ts |

### KV Cache Debugging

```typescript
// In attention.ts
console.log(`[ATT_DEBUG] Decode L${layerIdx}: seqLen=${currentSeqLen}, numTokens=${numTokens}`);
console.log(`[ATT_PARAMS] kvLenForAttention=${kvLenForAttention}, startPosForMask=${startPosForMask}`);

// Verify cache has data
const gpuBuffers = kvCache.getGPUBuffers(layerIdx);
console.log(`[KV] Layer ${layerIdx}: keysSize=${gpuBuffers.keysGPU.size}, seqLen=${gpuBuffers.seqLen}`);
```

---

## 6. RoPE Position Debugging

```typescript
// Verify RoPE frequencies precomputed for full context
console.log(`[RoPE] freqsCos size=${ropeFreqsCos.size}, expected=${maxSeqLen * headDim * 2}`);

// Verify startPos passed correctly
console.log(`[RoPE] Q startPos=${startPos}, numHeads=${numHeads}, headDim=${headDim}`);
```

### Architecture-Specific RoPE

| Model | Theta | Notes |
|-------|-------|-------|
| LLaMA | 10000 | Standard |
| Gemma 3 | 1000000 | Higher theta |
| GPT-OSS | 10000 | YARN scaling factor=32 |

---

## 7. Sampling & Logits Debugging

```typescript
// Before softmax
console.log(`[Logits] raw: min=${min}, max=${max}, range=${max-min}`);

// After softmax
const topK = getTopK(probs, 5);
console.log(`[Sample] top-5: ${topK.map(t => `${t.token}:${(t.prob*100).toFixed(1)}%`).join(', ')}`);

// Red flags
if (topK[0].prob < 0.05) console.warn("Near-uniform distribution - signal destroyed");
if (topK[0].prob > 0.99) console.warn("Overconfident - possible bug in earlier layers");
```

---

## 8. Browser-Specific Issues

### Clear All Caches

```javascript
// Clear localStorage
localStorage.clear();

// Clear Cache API
caches.keys().then(k => k.forEach(c => caches.delete(c)));

// Hard refresh (Cmd+Shift+R / Ctrl+Shift+R)

// For OPFS (model weights) - use demo UI "Clear Cache" button
// Or programmatically:
import { deleteModel } from './storage/shard-manager.js';
await deleteModel(modelId);
```

### HTTP Cache Bypass (for tests)

```typescript
// In test runner
await context.route('**/*', (route) => {
  route.continue({
    headers: {
      ...route.request().headers(),
      'Cache-Control': 'no-cache, no-store, must-revalidate',
    },
  });
});
```

---

## 9. Reference Comparison

### Compare Against llama.cpp

```bash
# Run same prompt through llama.cpp with debug output
./main -m model.gguf -p "the sky is" --n-gpu-layers 0 -n 5 --verbose

# Compare:
# - Token IDs produced
# - Layer activations (use --log-disable for quiet, --log-verbose for full)
```

### Compare Against transformers.js

```javascript
import { pipeline } from '@xenova/transformers';
const generator = await pipeline('text-generation', 'model-name');
const output = await generator('the sky is', { max_new_tokens: 5 });
```

---

## 10. Memory & Buffer Issues

### Common Memory Bugs

| Bug | Symptom | Fix |
|-----|---------|-----|
| Use-after-release | Zeros or corruption | Check releaseBuffer() timing |
| Wrong buffer size | Index out of bounds | Verify acquireBuffer() size |
| Buffer destroyed | WebGPU error | Ensure buffer lifetime spans usage |
| Dtype mismatch | Wrong values | Check setBufferDtype() calls |

### Debug Memory State

```typescript
// Track buffer pool state
import { getPoolStats } from './gpu/buffer-pool.js';
console.log('[Pool]', getPoolStats());
```

---

## 11. Test Commands

All CLI commands auto-start the server and run **headed (visible browser) by default**:

```bash
# Quick kernel validation (browser opens)
doppler test correctness

# Inference test with debug output
doppler bench inference --prompt xs --debug

# Layer-by-layer analysis
doppler bench inference --prompt xs --debug 2>&1 | grep -E "LAYER_[0-9]+_LAST"

# Final hidden state and logits
doppler bench inference --prompt xs --debug 2>&1 | grep -E "FINAL_HIDDEN|logits|top-5|Generated"

# Specific kernel test
doppler test correctness --filter matmul-q4k

# Specific model
doppler test inference --model gemma3-1b-q4

# Headless mode (for CI)
doppler test correctness --headless
doppler bench inference --prompt xs --headless
```

**Manual browser testing:** Run `npm start` first, then open `http://localhost:8080/d`.

---

## 12. Performance Debugging

### GPU Submit Tracking

DOPPLER includes submit tracking to measure per-token GPU overhead:

```typescript
import { setTrackSubmits, resetSubmitStats, logSubmitStats } from '../gpu/device.js';

// Enable tracking
setTrackSubmits(true);
resetSubmitStats();

// ... run forward pass ...

// Log results
logSubmitStats('Forward pass');
setTrackSubmits(false);
```

### Command Buffer Batching

**Before batching**: ~260+ GPU submits per forward pass (~50-100ms overhead)
**After batching**: 1 submit per forward pass (~0.5ms overhead)

The batching system uses `CommandRecorder` to record GPU operations into a single command buffer:

```typescript
import { createCommandRecorder } from '../gpu/command-recorder.js';

const recorder = createCommandRecorder('forward_pass');

// Use record* variants instead of run*
await recordMatmul(recorder, A, B, M, N, K);
await recordRMSNorm(recorder, input, weight, eps);

// Submit all at once
await recorder.submitAndWait();
```

### Key Files for Performance

| File | Debug Focus |
|------|-------------|
| `gpu/command-recorder.ts` | Batched command recording |
| `gpu/submit-tracker.ts` | GPU submit statistics |
| `inference/pipeline.ts` | Forward pass orchestration |
| `inference/pipeline/layer.ts` | do* wrappers for run/record variants |

---

## Postmortem Index

| Issue | Root Cause | File | Status |
|-------|-----------|------|--------|
| Garbage tokens (unused16) | Q4_K quantization format | [GEMMA3-DEBUG-POSTMORTEM.md](postmortems/GEMMA3-DEBUG-POSTMORTEM.md) | Fixed |
| FFN value explosion | Quantization + sign handling | [PIPELINE-VERIFICATION-POSTMORTEM.md](postmortems/PIPELINE-VERIFICATION-POSTMORTEM.md) | Fixed |
| Zero embeddings high token IDs | 2D dispatch linearization | [BF16-2D-DISPATCH-POSTMORTEM.md](postmortems/BF16-2D-DISPATCH-POSTMORTEM.md) | Fixed |
| Kernel outputs zeros | 'auto' layout mismatch | [MOE-EXPLICIT-LAYOUT-POSTMORTEM.md](postmortems/MOE-EXPLICIT-LAYOUT-POSTMORTEM.md) | Fixed |
| Decode broken, prefill works | SiLU gating bug | (this guide, Pattern A) | Fixed |
| Softmax test failure | Uniform buffer layout swapped | [SOFTMAX-UNIFORM-BUFFER-POSTMORTEM.md](postmortems/SOFTMAX-UNIFORM-BUFFER-POSTMORTEM.md) | Fixed |
| Hidden state explosion | q_norm/k_norm +1 offset + Q4K layout | [postmortems/INDEX.md](postmortems/INDEX.md) | Fixed |

---

## For Claude Agents

When debugging DOPPLER issues:

1. **Start with symptoms** - Use the Quick Diagnosis Table above
2. **Add logging** - Strategic console.log at pipeline stages
3. **Check value ranges** - maxAbs explosion is a red flag
4. **Verify shapes** - Buffer sizes must match expected dimensions
5. **Test boundaries** - Token IDs, sequence lengths, layer indices
6. **Check postmortems index** - See `docs/postmortems/INDEX.md` for common patterns and lessons learned
7. **Compare references** - llama.cpp or transformers.js as ground truth

### Key Files to Instrument

| File | Debug Focus |
|------|-------------|
| `inference/pipeline.ts` | Overall flow, token loop |
| `inference/pipeline/layer.ts` | Per-layer processing |
| `inference/pipeline/attention.ts` | KV cache, RoPE, attention |
| `gpu/kernels/silu.ts` | FFN activation gating |
| `gpu/kernel-selector.ts` | Kernel dispatch, buffer management |
| `loader/doppler-loader.ts` | Weight loading, dequantization |

---

*Last updated: 2025-12-29*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

