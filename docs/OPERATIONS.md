# DOPPLER Operations

## Troubleshooting

Comprehensive debugging strategies for DOPPLER WebGPU inference issues. Written for future developers and Claude agents.

**Note:** All CLI commands now run headless by default with real GPU (via `--headless=new`). Use `--headed` for visible browser window during debugging.
**Config-only overrides:** Prompt size, sampling, warmup/runs, trace, and log levels are configured via `runtime.shared.benchmark.run` and `runtime.shared.debug` in the runtime config passed to `--config`.

---

## Quick Start: Systematic Debug Workflow

### 1. Run Kernel Tests First
```bash
npm run doppler -- test correctness
```
If any kernel fails, **fix it first**. Expected: all PASS except scatter-add.

### 2. Run Inference Debug
```bash
npm run doppler -- bench inference --config debug
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

**Source formats:** DOPPLER's `node-converter` supports both **safetensors** (HuggingFace) and **GGUF** (llama.cpp) as input formats. The resulting RDRR format is the same regardless of source. If debugging issues, compare against the original source (HuggingFace for safetensors, llama.cpp for GGUF).

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
npm run doppler -- bench inference --config debug 2>&1 | grep -E "weight|norm.*min|norm.*max"
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

DOPPLER uses a unified logging system controlled by runtime config:

| Config | Level | Shows |
|--------|-------|-------|
| `runtime.shared.debug.logLevel.defaultLogLevel=info` | info | Phase starts/ends, totals |
| `...=verbose` | verbose | + Per-shard source, per-layer timing |
| `runtime.shared.debug.trace.enabled=true` | trace | + Tensor shapes, dequant ops, buffer details |
| `...=silent` | silent | Errors only |

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

The DOPPLER CLI (`cli/index.js`) runs Playwright and filters browser console logs before forwarding to stdout. Only logs with these tags are shown:

```
[Benchmark], [Pipeline], [Loader], [GPU], [Kernel], [Layer], [KERNEL], [KV], [ATTN], [FFN], ERROR, WARN
```

**If your logs don't appear:**
1. Check your grep pattern includes the tag (e.g., `Loader` to match loader output)
2. Use a config preset (e.g., `debug`) to enable verbose logging and trace
3. Some debug readbacks skip when using CommandRecorder (batched mode) - this is by design

```bash
# Show shard sources and layer timing
doppler bench --config debug 2>&1 | grep -E "Loader.*Shard|Loader.*Layer"

# Show everything including tensor details
doppler debug --config debug 2>&1 | head -200
```

### OPFS Cache Persistence (Faster Reruns)

The CLI uses a persistent Playwright profile directory to preserve browser storage between runs. This includes the OPFS model cache, so the second run should skip downloads.

- Default profile dirs:
  - Tests: `doppler/.test-cache/`
  - Inference benchmarks: `doppler/.benchmark-cache/`
- Override with `--profile-dir <path>` (relative to `doppler/` or absolute)

```bash
# Reuse the same profile across runs (warm OPFS)
doppler bench inference --config bench --profile-dir .benchmark-cache

# Use a fresh profile for a cold-start run
doppler bench inference --config bench --profile-dir .benchmark-cache-cold
```

### Log Format for Post-Filtering

All logs use a consistent format: `[CATEGORY][L{layer}][S{step}] message`

This enables grep-based filtering:

```bash
# Filter for specific categories
doppler bench inference --config bench 2>&1 | grep -E "^\[LOGITS\]"
doppler bench inference --config bench 2>&1 | grep -E "^\[LAYER\]\[L0\]"
doppler bench inference --config bench 2>&1 | grep -E "^\[ATTN\]|\[FFN\]"

# Watch layer 0 through decode steps
doppler bench inference --config bench 2>&1 | grep "\[L0\]" | head -20
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
// In layer.js, before/after each stage
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
grep -A 5 "uniformView.setUint32" gpu/kernels/softmax.js
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
# Test high token IDs (set runtime.shared.benchmark.run.customPrompt in config)
npm run doppler -- bench inference --config ./bench-token-10000.json
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
npm run doppler -- bench inference --config debug 2>&1 | grep "FFN.*down\|FFN.*FINAL"
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

**Fix**: Use `getNormWeightBuffer()` for q_norm/k_norm in attention.js. Reconvert model after loader fix.

---

## 4.1 Experimental Debug Techniques

### One-Liner Debug Scripts

```bash
# Watch hidden state explosion in real-time
npm run doppler -- bench inference --config debug 2>&1 | \
  grep -E "LAYER_[0-9]+.*maxAbs" | \
  while read line; do
    abs=$(echo "$line" | grep -oP 'maxAbs=[\d.]+' | cut -d= -f2)
    [ $(echo "$abs > 500" | bc -l) -eq 1 ] && echo "EXPLOSION: $line" || echo "$line"
  done

# Compare logit rankings for specific tokens
npm run doppler -- bench inference --config debug 2>&1 | \
  grep -E "blue=|BLUE=|sky=" | tail -5

# Extract just the layer-by-layer maxAbs values for plotting
npm run doppler -- bench inference --config debug 2>&1 | \
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
// Add to layer.js for surgical debugging
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
  npm run doppler -- bench inference --config debug 2>&1 | grep "top-5"
done
```

### Buffer Content Comparison

```bash
# Dump buffer contents for offline analysis
npm run doppler -- bench inference --config debug 2>&1 | \
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
| `startPosForMask` | `currentSeqLen` | Log in attention.js |

### KV Cache Debugging

```typescript
// In attention.js
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
doppler bench inference --config debug

# Layer-by-layer analysis
doppler bench inference --config debug 2>&1 | grep -E "LAYER_[0-9]+_LAST"

# Final hidden state and logits
doppler bench inference --config debug 2>&1 | grep -E "FINAL_HIDDEN|logits|top-5|Generated"

# Specific kernel test
doppler test correctness --filter matmul-q4k

# Specific model
doppler test inference --model gemma3-1b-q4

# Headless mode (for CI)
doppler test correctness --headless
doppler bench inference --config bench
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
| `gpu/command-recorder.js` | Batched command recording |
| `gpu/submit-tracker.js` | GPU submit statistics |
| `inference/pipeline.js` | Forward pass orchestration |
| `inference/pipeline/layer.js` | do* wrappers for run/record variants |

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
| `inference/pipeline.js` | Overall flow, token loop |
| `inference/pipeline/layer.js` | Per-layer processing |
| `inference/pipeline/attention.js` | KV cache, RoPE, attention |
| `gpu/kernels/silu.js` | FFN activation gating |
| `gpu/kernels/*.js` | Kernel selection and dispatch |
| `loader/doppler-loader.js` | Weight loading, dequantization |

---

*Last updated: 2025-12-29*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/style/WGSL_STYLE_GUIDE.md` for runtime kernel modes and the OPFS purge helper.


## Debug Sessions

**Last Updated**: 2025-12-17 23:13 UTC
**Status**: UNSOLVED - See [POSITIVE-BIAS-HIDDEN-STATES-POSTMORTEM.md](postmortems/POSITIVE-BIAS-HIDDEN-STATES-POSTMORTEM.md)

## Quick Start

1. **Invoke the doppler-debug skill** when investigating inference issues
2. **Run quick benchmarks** to reproduce:

```bash
# Reproduce the garbage output bug
doppler bench inference --config debug 2>&1 | grep -E "FINAL_HIDDEN|LAST_TOKEN|blue|Kaw|Generated"

# Look for: ALL POSITIVE values at last position
# FINAL_HIDDEN[pos=6]: [183.x, 42.x, 201.x, ...] - ALL POSITIVE (bug!)
```

## Current Issue (UNSOLVED)

**Prompt**: "The color of the sky is"
**Expected**: "blue"
**Actual**: "Kaw" (garbage token 44821)

| Observation | Value | Status |
|-------------|-------|--------|
| FINAL_HIDDEN[pos=0] | [-97, -21, -76, ...] Mixed | Correct |
| FINAL_HIDDEN[pos=6] | [183, 42, 201, ...] ALL POSITIVE | **BUG** |
| Token "Kaw" logit | 28.35 (MAX) | Wrong |
| Token "blue" logit | 4.81 | Should be higher |

**Root cause**: Hidden states at last token position are all positive, causing tokens with positive embeddings to dominate.

## Priority Investigation

1. **Q4_K dequantization verification** - Does GPU kernel actually produce negative values?
2. **Layer-by-layer tracking at pos=N-1** - When does positive bias start?
3. **Reference comparison** - Run llama.cpp with same weights and compare

See postmortem for full hypothesis ranking and next steps.

## Log Levels (verbosity)

Control general log verbosity via runtime config:

| Config | Level | Shows |
|--------|-------|-------|
| `runtime.shared.debug.logLevel.defaultLogLevel=info` | info | Phase starts/ends, totals |
| `...=verbose` | verbose | + Per-shard source (RAM/OPFS/network), per-layer timing |
| `...=silent` | silent | Errors only |

## Trace (categories)

Trace is separate from log level. Use runtime config for tensor/kernel details:

- `runtime.shared.debug.trace.enabled=true`
- `runtime.shared.debug.trace.categories=["kernels","attn"]`
- `runtime.shared.debug.trace.categories=["all","-buffers"]`

**Defaults by preset:**
- `bench`: log=info, trace off
- `debug`: log=verbose, trace on (all categories)

**Config-only usage:** `runtime.shared.debug.trace` is the source of truth.

## Config-Driven Probes (Preferred for Readbacks)

Use probes to read specific token/dimension values without adding ad-hoc logs:

```bash
# Post-softcap logits probe (Gemma 2 parity)
npm run debug -- --config '{
  "runtime": {
    "shared": {
      "debug": {
        "trace": { "enabled": true, "categories": ["logits"] },
        "probes": [
          { "id": "topk", "stage": "logits_final", "tokens": [-1], "dims": [476, 3868] }
        ]
      }
    }
  }
}'
```

Probes run on CPU or GPU buffers; they are skipped when CommandRecorder batching is active.

### Common Grep Patterns

```bash
# Debug with verbose loader output
doppler debug 2>&1 | grep -E "Loader.*Shard|Loader.*Layer" | head -50

# Layer-by-layer debug output
doppler debug 2>&1 | grep -E "Layer[0-9]" | head -50

# Full debug with logits and generated text
doppler debug 2>&1 | grep -E "Layer|logits|top-5|Generated" | head -50

# Position-specific hidden state debug
doppler debug 2>&1 | grep -E "FINAL_HIDDEN|LAST_TOKEN" | head -20

# Trace-level output (tensor details)
doppler debug --config debug 2>&1 | head -200
```

**If logs don't appear:** Check your grep pattern includes the tag (e.g., `Loader` for loader output).

## Debug Flag

**IMPORTANT:** Debug GPU readbacks are gated behind `runtime.shared.benchmark.run.debug=true` or the `debug` preset to avoid performance impact.

- Without flag: Benchmarks run at full speed (no GPU sync points)
- With flag: Verbose layer-by-layer output but much slower

```bash
# Fast benchmark (no debug output)
doppler bench inference --config bench --headed

# Slow benchmark with debug GPU readbacks
doppler bench inference --config debug
```

```typescript
// Programmatic debug
await pipeline.generate(prompt, { debug: true, maxTokens: 10 });
```

## Selective Layer Debugging (Faster)

Use `debugLayers` to debug only specific layers while keeping batching enabled for other layers:

```typescript
// Full debug (slow): syncs at EVERY layer
await pipeline.generate(prompt, { debug: true });

// Selective debug (faster): syncs only at checkpoint layers
await pipeline.generate(prompt, {
  debug: true,
  debugLayers: [0, 12, 25],  // First, middle, last layers
});
```

This dramatically speeds up debug runs by:
1. Keeping CommandRecorder enabled for non-checkpoint layers
2. Only flushing GPU commands and reading back hidden states at specified layers
3. Recreating the recorder after each checkpoint to continue batching

For Gemma 3 1B (26 layers), typical checkpoint choices:
- `[0]` - Only first layer (embedding issues)
- `[25]` - Only final layer (pre-logits state)
- `[0, 12, 25]` - First, middle, last (balanced)
- `[0, 1, 2, ..., 25]` - All layers (same as `debug: true` alone)

## OPFS Cache Persistence (Faster Reruns)

The benchmark runs inside a persistent Playwright profile directory. This preserves browser storage between runs, including the OPFS model cache.

- Default inference benchmark profile: `doppler/.benchmark-cache/`
- Override with `--profile-dir <path>` (relative to `doppler/` or absolute)

```bash
# Warm run (reuse existing OPFS cache)
doppler bench inference --config bench --profile-dir .benchmark-cache

# Cold run (fresh profile dir)
doppler bench inference --config bench --profile-dir .benchmark-cache-cold
```

## CommandRecorder Gotcha

**CRITICAL**: When using CommandRecorder (batched mode), debug readbacks show zeros!

Always check `!recorder` before attempting debug buffer reads:
```typescript
if (layerIdx === 0 && !recorder) {
  // Safe to debug readback
} else if (recorder) {
  console.log('(skipping - batched mode)');
}
```

## Key Files

- `inference/pipeline.js` - decode loop
- `inference/pipeline/layer.js` - layer processing with debug readbacks
- `inference/pipeline/logits.js` - final norm + lm_head
- `gpu/kernels/matmul_q4_fused.wgsl` - Q4_K dequantization kernel
- `inference/pipeline/probes.js` - config-driven probe readbacks
- `config/schema/debug.schema.js` - trace/probe schema

## Performance Context

Target: 40+ tok/s decode on Gemma 3 1B. See `feature-log/doppler/inference.jsonl` for task tracking.

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/style/WGSL_STYLE_GUIDE.md` for runtime kernel modes and the OPFS purge helper.


## Performance Regression Investigations

## Overview

**Current Performance:** ~4 tok/s
**Target Performance:** 40+ tok/s
**Gap:** 10x slower than target

This document investigates the performance bottlenecks causing the 10x performance gap and provides actionable fixes.

## Executive Summary

The 10x performance gap is caused by:
1. **Kernel launch overhead** - Too many small kernel dispatches per token
2. **Memory bandwidth saturation** - Redundant buffer reads/writes between kernels
3. **Suboptimal decode kernels** - Not leveraging M=1 optimizations
4. **Missing kernel fusion** - Separate gate/up projections waste bandwidth

## Profiling Methodology

### GPU Timing

Use the kernel benchmark harness to measure per-kernel timings:

```bash
# Kernel microbenchmarks
doppler test kernels --perf

# Full inference benchmark (for tok/s + latency)
doppler test inference --perf
```

### Expected Breakdown (Gemma 3 1B)

| Operation | Expected (ms) | Current (ms) | Notes |
|-----------|---------------|--------------|-------|
| RMSNorm (x2) | 0.02 | 0.2 | 10x overhead |
| QKV Matmul | 0.1 | 0.5 | Not using GEMV |
| Attention | 0.2 | 2.0 | Full softmax recompute |
| Out Proj | 0.05 | 0.3 | Not using GEMV |
| FFN Gate+Up | 0.15 | 1.5 | Separate kernels |
| FFN Down | 0.1 | 0.5 | Not using GEMV |
| LM Head | 0.5 | 3.0 | Large vocab (262K) |
| **Total/layer** | ~1.1 | ~8.0 | **7x gap** |
| **26 layers** | ~28 | ~208 | |
| **tok/s** | ~35 | ~5 | **7x gap** |

## Identified Bottlenecks

### 1. Kernel Launch Overhead (30% of gap)

**Problem:** Each decode token requires ~200+ kernel dispatches.

**Evidence:**
- RMSNorm: 2 per layer = 52 dispatches
- Matmul: 4 per layer = 104 dispatches
- Attention: 1 per layer = 26 dispatches
- Activation: 1 per layer = 26 dispatches
- Total: ~208 dispatches per token

**Fix:** Use CommandRecorder to batch all layer operations into a single submit.

```typescript
// Before: Multiple submits
for (const layer of layers) {
  await runRMSNorm(...);  // submit
  await runMatmul(...);   // submit
  await runAttention(...); // submit
}

// After: Single submit
const recorder = new CommandRecorder(device);
for (const layer of layers) {
  await recordRMSNorm(recorder, ...);
  await recordMatmul(recorder, ...);
  await recordAttention(recorder, ...);
}
recorder.submit();
```

**Expected Improvement:** 1.5-2x

### 2. Memory Bandwidth Waste (25% of gap)

**Problem:** Separate gate/up projections read input twice.

**Evidence:**
```
FFN gate: Read input (1152 * 4 bytes) + Read W_gate + Write output
FFN up:   Read input (1152 * 4 bytes) + Read W_up + Write output
Total: 2x input reads, 2x output writes
```

**Fix:** Use fused FFN kernel (`ffn_fused.wgsl`).

```typescript
// Before
const gate = await runMatmul(input, W_gate, 1, 6912, 1152);
const up = await runMatmul(input, W_up, 1, 6912, 1152);
const activated = await runSiLU(up, { gate });

// After
const activated = await runFusedFFN(input, W_gate, W_up, 1152, 6912, {
  activation: 'silu'
});
```

**Expected Improvement:** 1.3-1.5x for FFN pass

### 3. Non-optimized Decode Matmul (20% of gap)

**Problem:** Using batched matmul kernels for M=1 decode.

**Evidence:**
- Generic matmul: 16x16 tiles, many idle threads for M=1
- GEMV kernel: 256 threads, optimized for single-row

**Fix:** Ensure matmul.js selects GEMV variant for M=1:

```typescript
// In selectMatmulVariantAndFlags
if (M === 1 && effectiveBDtype === 'f16' && aDtype === 'f32') {
  if (capabilities.hasSubgroups) {
    variant = 'gemv_subgroup_multicol';  // For large N
  } else {
    variant = 'gemv';
  }
}
```

**Expected Improvement:** 2-3x for projections

### 4. Attention Decode Overhead (15% of gap)

**Problem:** Using prefill-style attention for single-token decode.

**Evidence:**
- Prefill kernel: Tiles over seqLen, unnecessary for seqLen=1
- Full softmax stored: Wastes shared memory

**Fix:** Use optimized decode attention kernel (`attention_decode_optimized.wgsl`):

- Online softmax (no full score storage)
- Vectorized KV cache reads
- Subgroup reductions

**Expected Improvement:** 3-4x for attention

### 5. LM Head Bottleneck (10% of gap)

**Problem:** 262K vocab matmul dominates decode time.

**Evidence:**
- LM head: 1152 x 262144 = 302M MACs per token
- Memory read: 262144 * 1152 * 2 bytes (F16) = 604MB

**Fix:**
1. Use multi-column GEMV (`gemv_subgroup_multicol`)
2. Consider top-k logit computation (skip full softmax)
3. Weight-tied embeddings can share memory

**Expected Improvement:** 1.5-2x

## Implementation Checklist

### Phase 1: Quick Wins (Day 1)

- [x] Ensure GEMV kernel selection for M=1 matmuls
- [x] Enable CommandRecorder for batched execution
- [x] Verify subgroup operations are being used

### Phase 2: Kernel Fusion (Day 2-3)

- [x] Implement fused FFN kernel
- [x] Integrate into pipeline
- [x] Benchmark before/after

### Phase 3: Attention Optimization (Day 4-5)

- [x] Implement optimized decode attention
- [x] Add online softmax
- [x] Benchmark against baseline

### Phase 4: LM Head (Day 6)

- [ ] Profile LM head performance
- [ ] Implement weight-tied optimization
- [ ] Consider partial logit computation

## Profiling Commands

```bash
# Run kernel benchmarks
doppler test kernels --perf

# Full inference benchmark
doppler test inference --perf

# Enable kernel trace logging
doppler test inference --config debug
```

## Success Metrics

| Metric | Current | Target | Achieved |
|--------|---------|--------|----------|
| tok/s | 4 | 40+ | TBD |
| Per-layer latency | 8ms | 1ms | TBD |
| Total decode latency | 200ms | 25ms | TBD |

## Appendix: Kernel Timings Reference

### Gemma 3 1B (M1 Pro, WebGPU)

Expected optimal timings:
- RMSNorm (1x1152): 0.01ms
- MatMul GEMV (1x1152x1152): 0.05ms
- MatMul GEMV (1x6912x1152): 0.1ms
- MatMul GEMV (1x262144x1152): 0.5ms
- Attention decode (4 heads, 256 dim, 512 KV): 0.1ms
- SiLU (6912 elements): 0.01ms

Total per layer: ~0.4ms
26 layers + LM head: ~11ms
tok/s: ~90

With overhead and non-optimal paths: ~40 tok/s target is achievable.


## Test Results

Index of DOPPLER validation sessions across different hardware and browsers.

## Quick Status

| Platform | Status | Last Tested |
|----------|--------|-------------|
| Apple M3 (macOS) | Working | Dec 2025 |
| AMD Strix Halo (Linux) | Blocked (headless) | Dec 2025 |
| NVIDIA | Untested | - |

---

This file is a human-readable log. Store machine-readable benchmark outputs as JSON using
`docs/style/BENCHMARK_STYLE_GUIDE.md` so results can be compared automatically.

See also:
- `docs/style/BENCHMARK_STYLE_GUIDE.md` for benchmark methodology and JSON result schema
- `docs/design/KERNEL_TESTING.md` for WGSL kernel testing specification
- `tests/kernels/README.md` for kernel test coverage
- `tests/kernels/BENCHMARKS.md` for kernel microbenchmark baselines

## Result Artifacts (Recommended)

| Artifact | Purpose | Suggested Path |
|----------|---------|----------------|
| Pipeline benchmark JSON | TTFT, tok/s, submits, readback, memory | `tests/results/` |
| Kernel correctness JSON/HTML | per-kernel correctness | `tests/kernels/test-results/` |
| Kernel benchmark JSON/HTML | per-kernel timings | `tests/kernels/test-results/` |
| Baseline registry | Expected tok/s ranges | `tests/baselines.json` |

If a run does not have a JSON artifact yet, record the session here and file it as follow-up work.

## Test Sessions

### Session 2025-12-14: AMD Strix Halo + Gemma 3 1B

**Tester**: Linux AMD machine
**GPU**: AMD Strix Halo integrated GPU (Radeon 8050S/8060S Graphics)
**Browser**: Google Chrome 142.0.7444.162
**OS**: Linux 6.17.0-7-generic
**Model**: Gemma 3 1B IT (Q4_K_M quantization)

#### Status: IN PROGRESS

**Steps completed**:
1. ✓ Located Gemma 3 1B model in HuggingFace cache
2. ✓ Converted to RDRR format (Q4_K_M quantization) - 965MB, 15 shards
3. ✓ Model served locally for browser load
4. ☒ **BLOCKED**: Headless browser cannot access WebGPU (no GPU in headless environment)

**Test Limitation**: The Linux environment runs headless without X server or GPU access. WebGPU requires either:
- A headed browser with GPU drivers (X11/Wayland + working GPU)
- OR Manual testing in a desktop environment

**Model is ready** - just needs a desktop browser to test.

**Model path**:
- Source: `/home/clocksmith/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- RDRR output: `./models/gemma-3-1b-q4/`

**Expected model size**: ~1.2GB (340 tensors, 26 layers, 1152 hidden size)

---

### Session 2025-12-14: Mac M3 + GPT-OSS 20B (parallel session)

**Tester**: MacBook with M3
**GPU**: Apple M3 (unified memory)
**Model**: GPT-OSS 20B MoE (Q4_K_M, 32 experts, topK=4)
**Status**: PARTIAL - Router fixed, expert loading in progress

#### Bug Fix 1: MoE Gather Kernel (FIXED)
- Root cause: WebGPU `layout: 'auto'` only includes bindings used by each entry point
- `count_and_map` used 4/6 bindings, `gather_tokens` used 6/6
- Bind group creation with mismatched layout caused silent failure
- Fix: Created explicit bind group layout with all 6 bindings
- See: `docs/MOE-EXPLICIT-LAYOUT-POSTMORTEM.md`

#### Bug Fix 2: Router Weight Quantization (FIXED)
- Root cause: Router weights quantized to Q4_K_M despite HuggingFace config `modules_to_not_convert`
- Symptom: Router logits extreme (56 vs -39), softmax collapses to [1.0, 0.0, 0.0, 0.0]
- Fix: Updated `shouldQuantize()` in quantizer.js to check:
  1. Hard-coded `router` and `gate.weight` patterns
  2. HuggingFace `modules_to_not_convert` config from quantization_config
- Reconverted model: Router weights now BF16 (184KB vs 52KB Q4_K_M)
- **Result**: Router now produces distributed weights!
  ```
  [DEBUG MoE L0] Router logits (first 8 experts): -0.14, 0.59, 0.11, 0.88, -0.98, -1.73, 2.58, -0.54
  [DEBUG MoE L0] Expert weights: [0.5896, 0.1668, 0.1359, 0.1078, ...]
  ```
  vs before: `[1.0, 0.0, 0.0, 0.0]`

#### Current Status: Expert Loading
- Router works correctly (distributed weights)
- Expert tensor loading attempted but using wrong naming convention
- GPT-OSS uses packed MXFP4 experts (`model.layers.X.mlp.experts.gate_up_proj_blocks`)
- Loader fallback exists but may need debugging

**Files modified**:
- `gpu/kernels/moe.js` - Added explicit bind group layout for MoE
- `gpu/kernels/moe_gather.wgsl` - Cleaned up, added layout note
- `src/converter/quantizer.js` - Added router check in `shouldQuantize()`
- `src/converter/node-converter.js` - Pass `modules_to_not_convert` to shouldQuantize

---

## Hardware Configurations Tested

| Date | GPU | VRAM | Browser | OS | Model | Status | Notes |
|------|-----|------|---------|----|----|-------|-------|
| 2025-12 | Apple M3 | Unified | Safari/Chrome | macOS | Gemma 3 1B | ✓ WORKING | Reference implementation |
| 2025-12-14 | AMD Strix Halo | Integrated | Chrome 142 | Linux | Gemma 3 1B | ⏳ TESTING | In progress |
| 2025-12-14 | Apple M3 | Unified | Chrome | macOS | GPT-OSS 20B | ⚠️ PARTIAL | MoE pipeline works, output quality poor |

## Test Protocol

### Standard Result Capture (Recommended)

For each performance session, record:

- Model: `modelId`, quantization, shard count, tensor count
- Environment: OS, browser version, GPU adapter info, WebGPU feature flags
- Workloads: prompt names and token counts
- Metrics: TTFT, prefill tok/s, decode tok/s, peak VRAM estimate, GPU submit counts
- Output quality: `quality.ok` plus reasons/warnings

Preferred output:

- A JSON file per run matching `docs/style/BENCHMARK_STYLE_GUIDE.md`
- A short narrative summary in this document for context and troubleshooting.
Baseline ranges live in `tests/baselines.json` and are enforced in CI when enabled.

To avoid instruction drift, prefer linking to the canonical runner docs:

- Kernel tests and microbenchmarks: `tests/kernels/README.md` and `tests/kernels/BENCHMARKS.md`
- End-to-end inference tests: `tests/harness.html` (set `runtime.shared.harness.mode` via `runtimeConfig`)

## Known Issues by Platform

### AMD GPUs
- **Driver requirements**: Mesa 23.0+ (Linux) or Adrenalin 23.0+ (Windows)
- **WebGPU status**: Generally good support in recent drivers
- **Strix Halo**: New integrated RDNA architecture, untested

### Apple Silicon
- **Unified memory advantage**: No PCIe overhead, can load larger models
- **Safari vs Chrome**: Both support WebGPU, Safari may have better integration
- **F16 support**: Excellent on M-series chips

### NVIDIA
- **Status**: Untested in DOPPLER
- **Expected**: Should work well with recent drivers
- **Driver**: 525+ required for WebGPU

## Debugging Common Issues

### Model loads but produces garbage tokens
**Symptom**: Output like `<unused16>`, random Unicode, or non-English text for English prompts

**Causes**:
1. Quantization format mismatch (Q4_K encoding issue)
2. BF16 conversion error
3. Gemma 3 norm offset not applied
4. GPU dequantization kernel bug

**Debug**:
- Check logs for "Prefill logits" top-5 distribution
- Look for negative hidden state values (should be present)
- Compare against known-working Mac M3 output

### WebGPU not available
**Symptom**: `navigator.gpu` is undefined

**Solutions**:
- Update browser (Chrome 113+, Edge 113+)
- Enable in Firefox: `about:config` → `dom.webgpu.enabled`
- Check GPU drivers are up to date

### Out of memory errors
**Symptom**: Buffer allocation fails, model won't load

**Solutions**:
- Try smaller model (Gemma 3 1B needs ~1.2GB)
- Close other GPU-intensive apps
- Check browser console for specific buffer size limits

## Contributing Results

After testing:
1. Update this file with your results
2. Update HARDWARE_COMPATIBILITY.md matrix
3. Commit and push changes
4. Share any issues or findings

---

*Last updated: January 2026*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/style/WGSL_STYLE_GUIDE.md` for runtime kernel modes and the OPFS purge helper.
