# DOPPLER Operations

## Troubleshooting

Comprehensive debugging strategies for DOPPLER WebGPU inference issues. Written for future developers and Claude agents.

**Note:** Doppler is browser-only. Use the demo diagnostics UI or `tests/harness.html`
with `runtimeConfig` to run suites. Runtime behavior is controlled via `runtime.*`
and `runtime.shared.tooling.intent`.

---

## Quick Start: Systematic Debug Workflow

### 1. Run Kernel Tests First
Open `tests/harness.html` in `kernels` mode (runtime config sets `runtime.shared.tooling.intent = "verify"`).
If any kernel fails, **fix it first**. Expected: all PASS except scatter-add.

### 2. Run Inference Debug
Use the demo diagnostics UI with the `debug` runtime preset and run the inference suite.

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

**Source formats:** DOPPLER's browser converter supports both **safetensors** (HuggingFace) and **GGUF** (llama.cpp) as input formats. The resulting RDRR format is the same regardless of source. If debugging issues, compare against the original source (HuggingFace for safetensors, llama.cpp for GGUF).

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

Use the demo diagnostics UI with a debug preset and inspect the DevTools console.
Filter logs for `weight` or `norm` to spot range anomalies.

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

Open `tests/harness.html` in `kernels` mode and call the dequant helpers from the console:
`window.testHarness.runDequantQ4K(...)` or `window.testHarness.runDequantAndMatmulF16W(...)`.

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

### Browser Log Filtering

All logs stay in the browser console. Use DevTools filtering or the log history API.

Example DevTools filters:

- `/Loader.*(Shard|Layer)/`
- `/^\\[LOGITS\\]/`
- `/^\\[LAYER\\]\\[L0\\]/`
- `/^\\[(ATTN|FFN)\\]/`

Example log history slicing:

```javascript
const logs = DOPPLER.getLogHistory();
const hits = logs.filter((entry) => entry.message?.includes('Loader'));
console.log(hits.slice(0, 200));
```

### OPFS Cache Persistence

OPFS is persisted per browser profile. For warm runs, reuse the same profile or
keep a tab open. For cold runs, clear OPFS or use a fresh profile/incognito.

```javascript
const { listModels, deleteModel } = await import('../src/storage/shard-manager.js');
for (const modelId of await listModels()) {
  await deleteModel(modelId);
}
```

### Log Format for Post-Filtering

All logs use a consistent format: `[CATEGORY][L{layer}][S{step}] message`. Use
DevTools filters or `DOPPLER.getLogHistory()` to slice by category/layer.

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

These patterns are consolidated from actual debugging sessions. Detailed
postmortems are tracked internally.

### Pattern A: Uniform Buffer Layout Mismatch
**Postmortem**: Internal - SOFTMAX-UNIFORM-BUFFER

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
**Postmortem**: Internal - GEMMA3-DEBUG

**Symptom**: All dequantized values positive, no negative weights.

**Root cause**: Quantizer stores `min` differently than llama.cpp format. Must store `-actual_min` as positive offset.

**Quick check**: Open `tests/harness.html` in `kernels` mode and run the
dequant round-trip from the console:
`window.testHarness.runDequantQ4K(...)` or `window.testHarness.runDequantAndMatmulF16W(...)`.

**Fix**: `value = d * sc * q - dmin * min` (subtract, not add)

### Pattern C: 2D Dispatch Without Linearization
**Postmortem**: Internal - BF16-2D-DISPATCH

**Symptom**: Works for small tensors, zeros/garbage for large tensors (>65K workgroups).

**Root cause**: Kernel ignores `global_id.y` in 2D dispatch. WebGPU limits 65535 workgroups per dimension.

**Quick check**: Run inference or bench mode with a long prompt. Set
`runtime.shared.benchmark.run.customPrompt` (or `runtime.inference.prompt`)
to a 10k-token string and launch `tests/harness.html` in `bench` or `inference`
mode.

**Fix**:
```wgsl
let linear_idx = global_id.y * (uniforms.workgroupsX * WORKGROUP_SIZE) + global_id.x;
```

### Pattern D: 'auto' Layout Silent Failure
**Postmortem**: Internal - MOE-EXPLICIT-LAYOUT

**Symptom**: Kernel runs without errors but outputs all zeros.

**Root cause**: `layout: 'auto'` with multi-entry-point shaders. WebGPU silently ignores binding mismatches.

**Quick check**: Create minimal test kernel with single binding to isolate.

**Fix**: Always use explicit bind group layout for complex shaders:
```typescript
const layout = device.createBindGroupLayout({ entries: [/* ALL bindings */] });
```

### Pattern E: FFN Value Explosion (Masked by Sandwich Norm)
**Postmortem**: Internal - PIPELINE-VERIFICATION

**Symptom**: Near-uniform logits (<10% top token probability).

**Root cause**: FFN explodes but post-FFN norm masks it. Information already destroyed.

**Quick check**: Enable `runtime.shared.debug.trace` with categories `ffn`
and filter the console with `/FFN.*(down|FINAL)/`. Values > 1000 indicate
an explosion.

**Fix**: Track values at every stage BEFORE normalization.

### Pattern F: Hidden State Explosion
**Postmortem**: Internal index (q_norm/k_norm and Q4K sections)

**Symptom**: maxAbs grows from ~20 to 800+ through layers. Output is garbage Unicode.

**Root cause**: q_norm/k_norm weights missing +1 offset (Gemma 3 uses `(1 + weight)` formula for ALL norms), combined with Q4K layout mismatch causing fallback to dequantized weights.

**Quick check**: Enable `runtime.shared.debug.trace` and filter the console
with `/TRACE|explosion/` or slice `DOPPLER.getLogHistory()`.

**Fix**: Use `getNormWeightBuffer()` for q_norm/k_norm in attention.js. Reconvert model after loader fix.

---

## 4.1 Experimental Debug Techniques

### Browser Debug Helpers

```javascript
// Watch hidden state maxAbs values
const logs = DOPPLER.getLogHistory({ last: 500 });
const maxAbs = logs
  .filter((entry) => entry.message.includes('maxAbs='))
  .map((entry) => entry.message);
console.log(maxAbs);

// Compare recent top-5 logits
const top5 = logs
  .filter((entry) => entry.message.includes('top-5'))
  .slice(-5);
console.log(top5.map((entry) => entry.message));
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

Use `runtime.shared.debug.trace.layers` or `runtime.shared.debug.pipeline.layers`
to focus logs on specific layers. If you must hard-stop execution at a layer,
add a temporary guard in the pipeline and reload the browser.

### Buffer Content Comparison

Use `DOPPLER.getLogHistory()` to collect buffer readback logs and copy them
from the DevTools console for offline diffing.

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
import { getPoolStats } from './memory/buffer-pool.js';
console.log('[Pool]', getPoolStats());
```

---

## 11. Test Runs (Browser)

All test flows run via `tests/harness.html` or the demo UI. Set
`runtime.shared.tooling.intent` to match the workload (verify/investigate/calibrate).

Example configs:

```json
{
  "shared": {
    "tooling": { "intent": "verify" },
    "harness": { "mode": "kernels", "autorun": true }
  }
}
```

```json
{
  "shared": {
    "tooling": { "intent": "verify" },
    "debug": { "logLevel": { "defaultLogLevel": "verbose" } },
    "harness": { "mode": "inference", "autorun": true, "modelId": "gemma3-1b-q4" }
  },
  "inference": { "prompt": "Hello from Doppler." }
}
```

Use DevTools filters for layer-by-layer analysis (e.g. `/LAYER_.*_LAST/`) and
top-5 logits (e.g. `/top-5/`). For specific kernel checks, call
`window.testHarness.runMatmul(...)` or related helpers from the console.

**Manual browser testing:** Run `python3 -m http.server 8080`, then open
`http://localhost:8080/tests/harness.html` or `http://localhost:8080/demo/`.

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
| Garbage tokens (unused16) | Q4_K quantization format | GEMMA3-DEBUG (internal) | Fixed |
| FFN value explosion | Quantization + sign handling | PIPELINE-VERIFICATION (internal) | Fixed |
| Zero embeddings high token IDs | 2D dispatch linearization | BF16-2D-DISPATCH (internal) | Fixed |
| Kernel outputs zeros | 'auto' layout mismatch | MOE-EXPLICIT-LAYOUT (internal) | Fixed |
| Decode broken, prefill works | SiLU gating bug | (this guide, Pattern A) | Fixed |
| Softmax test failure | Uniform buffer layout swapped | SOFTMAX-UNIFORM-BUFFER (internal) | Fixed |
| Hidden state explosion | q_norm/k_norm +1 offset + Q4K layout | Internal postmortem index | Fixed |

---

## For Claude Agents

When debugging DOPPLER issues:

1. **Start with symptoms** - Use the Quick Diagnosis Table above
2. **Add logging** - Strategic console.log at pipeline stages
3. **Check value ranges** - maxAbs explosion is a red flag
4. **Verify shapes** - Buffer sizes must match expected dimensions
5. **Test boundaries** - Token IDs, sequence lengths, layer indices
6. **Check postmortems index** - Internal postmortems cover common patterns and lessons learned
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
See `style/WGSL_style-guide.md` for runtime kernel modes and the OPFS purge helper.

