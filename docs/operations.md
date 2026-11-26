# Doppler Operations

## Troubleshooting

Comprehensive debugging strategies for Doppler WebGPU inference issues. Written for future developers and coding agents.

**Note:** Doppler supports browser and Node workflows. Use the demo diagnostics
UI or `tests/harness.html` for interactive/browser suites, and use the Node CLI
for scripted runs. Runtime behavior is controlled via `runtime.*`; active intent
is normalized from the command into `request.intent`.

## Quick Troubleshooting

- WebGPU unavailable: verify browser/runtime support, confirm adapter/device creation succeeds, and update GPU drivers if capability detection fails.
- Model loading failed: confirm `modelId`, `modelUrl`, manifest reachability, and shard availability before changing runtime config.
- Out of memory: use a smaller model or quantization, reduce workload size, and re-check [performance-sizing.md](performance-sizing.md) before treating it as a runtime bug.
- Slow performance: validate warm vs cold conditions, keep benchmark profile knobs consistent, and compare against [benchmark-methodology.md](benchmark-methodology.md) before publishing numbers.

---

## Quick Start: Systematic Debug Workflow

Follow the fail-closed ladder in the
[Inference Debug Protocol](agents/debug-protocol.md) and interpret each step with
the [Debug Playbook](debug-playbook.md). Two Doppler-specific first steps:

1. Run the `verify` command with workload `kernels` (command metadata normalizes
   `request.intent = "verify"`). Fix any failing kernel first. If `scatter_add`
   is the only failure, treat it as a targeted MoE kernel issue.
2. Run the inference workload in the demo diagnostics UI with the
   `profiles/verbose-trace` runtime profile.

For the trusted reference, see [Reference Comparison](#9-reference-comparison).

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

Use the demo diagnostics UI with the `profiles/verbose-trace` profile and inspect the DevTools console.
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

// Compare with the source tokenizer, e.g. Transformers.js:
import { AutoTokenizer } from '@huggingface/transformers';
const reference = await AutoTokenizer.from_pretrained('google/gemma-3-1b-it');
console.log("Reference tokens:", reference.encode("The color of the sky is"));
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

Always verify buffer sizes match expected dimensions. Enable `runtime.shared.debug.trace.enabled=true` to capture tensor shapes, dequant ops, and buffer details automatically. See [Section 2](#2-toggleable-debug-categories) for trace categories and log levels.

For Node runs, pass a runtime profile or config that sets `runtime.shared.debug.trace`. In the browser, call `DOPPLER.setTrace(...)` (see [Browser Console API](#browser-console-api)). GPU readback inspection is `runtime.shared.debug.pipeline.bufferStats: true`.

### Matmul Dimension Checklist

Matmul computes `A[M,K] @ B[K,N] -> C[M,N]`. Verify:
- Input A: `[numTokens, inputDim]`
- Weight B: `[inputDim, outputDim]` (or transposed)
- Output C: `[numTokens, outputDim]`

Wrong `transposeB` flag causes silent corruption.

---

## 2. Toggleable Debug Categories

DOPPLER has two category-filtered debug systems, both driven by runtime config:

- **Trace** (`runtime.shared.debug.trace`): `enabled`, `categories`, `layers`,
  `maxDecodeSteps`. The browser console can change it at run time.
- **Pipeline debug** (`runtime.shared.debug.pipeline`): `enabled`,
  `categories`, `layers`, `maxDecodeSteps`, `maxAbsThreshold`, `bufferStats`,
  `readbackSampleSize`. It is read from runtime config when the text pipeline
  initializes; the console global does not change it.

### Browser Console API

The `DOPPLER` console global exists only after an application entry point calls
`installDebugGlobal()`, exported from `dopplerx/tooling` and
`dopplerx/tooling/runtime`. The demo calls it in `demo/demo.js`. It returns a
function that uninstalls the global. The full surface is `DOPPLER_API` in
`src/debug/index.js`.

```javascript
// In browser DevTools console, after installDebugGlobal():
DOPPLER.setTrace('attn,ffn')                       // enable trace categories
DOPPLER.setTrace('attn,ffn', { layers: [0, 25] })  // only these layers
DOPPLER.setTrace('all,-buffers', { maxDecodeSteps: 5 })
DOPPLER.getTrace()                                 // enabled categories
DOPPLER.setTrace(false)                            // disable tracing
DOPPLER.setLogLevel('verbose')                     // debug | verbose | info | warn | error | silent
DOPPLER.getLogHistory({ module: 'Loader', last: 200 })
DOPPLER.printLogSummary(20)
DOPPLER.getDebugSnapshot()
await DOPPLER.inspect(gpuBuffer, 'label', { shape: [rows, cols] })
```

### Trace Categories

`runtime.shared.debug.trace.categories` and `DOPPLER.setTrace()` accept
`loader`, `kernels`, `logits`, `embed`, `attn`, `ffn`, `kv`, `sample`,
`buffers`, `perf` and `energy` (`TRACE_CATEGORIES` in `src/debug/config.js`).
`all` enables every category and `-name` removes one. Unknown names throw.

### Pipeline Debug Categories

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
| `kernel` | Kernel output buffer inspection (with `bufferStats`) |
| `all` | Enable everything |

### Log Levels

`runtime.shared.debug.logLevel.defaultLogLevel` (or `DOPPLER.setLogLevel()`)
takes `debug`, `verbose`, `info`, `warn`, `error` or `silent`:

| Level | Shows |
|-------|-------|
| `info` (default) | Phase starts/ends, totals |
| `verbose` | + Per-shard source, per-layer timing |
| `debug` | + Debug messages |
| `warn`, `error`, `silent` | Progressively less |

Tensor shapes, dequant ops and buffer details come from tracing
(`runtime.shared.debug.trace.enabled=true`), not from a log level. The
`profiles/verbose-trace` runtime profile sets `verbose` and enables all trace
categories.

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

The `dopplerx/compat` facade (tier `experimental`; the root `dopplerx` entry is
the Capsule runtime and has no `dr`) populates and reopens the verified cache
explicitly:

```javascript
import { dr } from 'dopplerx/compat';

const model = await dr.load('qwen3-0.8b', {
  cache: 'opfs',
  onProgress: ({ phase, message }) => console.log(phase, message),
});

console.log(model.persistentCache);
```

`cache: 'opfs'` never silently degrades to HTTP-only loading. If OPFS is
unavailable, persistence or integrity validation fails and the load rejects.
Use `cache: false` or omit `cache` when streaming directly from the artifact
host is intentional.

To purge every stored model from DevTools on a repo checkout page (for example
`/tests/harness.html`):

```javascript
const { listModels, deleteModel } = await import('../src/storage/shard-manager.js');
for (const modelId of await listModels()) {
  await deleteModel(modelId);
}
```

### Export A Model From Local Storage (OPFS/IndexedDB)

If you need to move a converted/downloaded RDRR model out of browser storage,
use the export helper (streams bytes; avoids full-file RAM materialization).

Requires the File System Access API (`showDirectoryPicker`, Chromium-based).

```javascript
import { exportModelToDirectory } from 'dopplerx/tooling/storage';

const destRoot = await window.showDirectoryPicker({ mode: 'readwrite' });
const dest = await destRoot.getDirectoryHandle('gemma-3-1b-q4', { create: true });

await exportModelToDirectory('gemma-3-1b-q4', dest, {
  onProgress: (p) => console.log(p),
});
```

### Log Format for Post-Filtering

Pipeline debug logs use `[CATEGORY][L{layer}][S{step}] message`. Trace logs use
`[<ms>ms][TRACE:<category>] L<layer>:message`. Use DevTools filters or
`DOPPLER.getLogHistory()` to slice by category/layer.

### Pipeline Debug Options

```json
{
  "shared": {
    "debug": {
      "pipeline": {
        "enabled": true,
        "categories": ["layer", "attn"],
        "layers": [0, 1],
        "maxDecodeSteps": 5,
        "maxAbsThreshold": 10000,
        "bufferStats": true
      }
    }
  }
}
```

`layers` limits logging to those layers, `maxDecodeSteps` to the first N decode
steps, and `maxAbsThreshold` warns on value explosion. `bufferStats` enables GPU
buffer readback and is expensive.

Runtime profiles under `src/config/runtime/profiles/` and
`src/config/runtime/diagnostics/` (for example `profiles/verbose-trace`) bundle
these settings; there are no named console debug presets.

---

## 3. Pipeline Stage Debugging

### Use Config-Driven Trace Probes

To inspect buffer stats at specific pipeline stages, enable the trace system via runtime config rather than modifying source files.

Enable per-layer tracing in the runtime profile or config payload:

```json
{
  "shared": {
    "debug": {
      "trace": {
        "enabled": true,
        "categories": ["attn", "ffn", "embed"],
        "layers": [0]
      }
    }
  }
}
```

To flag value explosions, set `maxAbsThreshold` in the pipeline debug config:

```json
{
  "shared": {
    "debug": {
      "pipeline": {
        "enabled": true,
        "categories": ["layer", "attn", "ffn"],
        "maxAbsThreshold": 1000,
        "bufferStats": true
      }
    }
  }
}
```

The trace system emits min/max/mean/maxAbs statistics at each instrumented stage automatically. Do not add ad-hoc `console.log` calls to source files — use the existing trace categories, config-driven probes, or permanent trace extensions instead.

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

```javascript
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

```javascript
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
# Compare WGSL struct definition with JavaScript write order
grep -A 10 "struct.*Uniforms" src/gpu/kernels/softmax.wgsl
grep -n -e "view.setUint32" -e "view.setFloat32" src/gpu/kernels/softmax.js
```

**Fix**: Add comments documenting WGSL layout at every uniform write:
```javascript
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
```javascript
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

Compare per-layer activations against a trusted reference with semantic
boundary receipts (`doppler boundary capture` and `doppler boundary compare`);
see [Evidence Loop](evidence-loop.md#2-compare-semantic-boundaries). For
reference engines, see [Reference Comparison](#9-reference-comparison).

### Isolate Specific Layer

Use `runtime.shared.debug.trace.layers` to focus tracing on a specific layer. In the browser, pass `layers` to `DOPPLER.setTrace()`:

```javascript
// Browser DevTools console:
DOPPLER.setTrace('attn,ffn', { layers: [14] })     // Focus on layer 14 only
DOPPLER.setTrace('attn,ffn', { layers: [0, 14] })  // Focus on a specific set
```

Or via runtime config:

```json
{
  "shared": {
    "debug": {
      "trace": { "enabled": true, "categories": ["attn", "ffn"], "layers": [14] }
    }
  }
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
| `currentSeqLen` | Increments each step | Enable `runtime.shared.debug.trace` |
| `startPos` for RoPE | `currentSeqLen` (not 0) | Enable `attn` trace category |
| `kvLen` for attention | `currentSeqLen + numTokens` | Enable `attn` trace category |
| `startPosForMask` | `currentSeqLen` | Enable `attn` trace category |

### KV Cache Debugging

Enable the `attn` and `kv` trace categories to surface KV cache state per layer without modifying source files:

```json
{
  "shared": {
    "debug": {
      "trace": { "enabled": true, "categories": ["attn", "kv"], "layers": [0] }
    }
  }
}
```

In the browser: `DOPPLER.setTrace('attn,kv')`.

---

## 6. RoPE Position Debugging

Enable the `attn` trace category to capture RoPE frequency buffer sizes and `startPos` values per layer. In the browser: `DOPPLER.setTrace('attn')`.

For architecture-specific RoPE config, inspect the manifest fields `rope_theta`, `rope_local_base_freq`, `ropeRotaryDim`, and `ropeInterleaved`. These drive frequency buffer sizing and must match the model architecture.

### Architecture-Specific RoPE

| Model | Theta | Notes |
|-------|-------|-------|
| LLaMA | 10000 | Standard |
| Gemma 3 | 1000000 | Higher theta |
| GPT-OSS | 150000 | YARN scaling factor=32 (see explicit conversion/runtime config for the chosen artifact) |

---

## 7. Sampling & Logits Debugging

Enable the `logits` and `sample` trace categories to surface raw logit range and top-k token probabilities without modifying source files.

In the browser:

```javascript
DOPPLER.setTrace('logits,sample')
```

Via runtime config:

```json
{
  "shared": {
    "debug": {
      "trace": { "enabled": true, "categories": ["logits", "sample"] }
    }
  }
}
```

Red flags to watch for in the trace output:
- Near-uniform distribution (top-1 probability < 5%) — signal destroyed early in the pipeline
- Overconfident output (top-1 probability > 99%) — possible earlier-layer bug masking real distribution

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
// Or programmatically (application code):
import { deleteModel } from 'dopplerx/tooling/storage';
await deleteModel(modelId);
```

### HTTP Cache Bypass (for tests)

```javascript
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
import { pipeline } from '@huggingface/transformers';
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

The buffer pool is a repository module, not a package export. From DevTools on
a repo checkout page:

```javascript
// Track buffer pool state
const { getBufferPool } = await import('../src/memory/buffer-pool.js');
console.log('[Pool]', getBufferPool().getStats());
```

---

## 11. Test Runs (Browser)

All test flows run through the browser or Node command surface. The command
contract derives the sole active `request.intent`; runtime profiles may declare
`compatibleIntents` but cannot author active intent.

Example configs:

```json
{
  "request": { "workload": "kernels" },
  "run": { "surface": "browser" }
}
```

```json
{
  "shared": {
    "tooling": { "diagnostics": "on-failure" },
    "debug": { "logLevel": { "defaultLogLevel": "verbose" } },
    "harness": { "mode": "inference", "autorun": true, "modelId": "gemma-3-270m-it-q4k-ehf16-af32" }
  },
  "inference": { "prompt": "Hello from Doppler." }
}
```

Use DevTools filters for layer-by-layer analysis (e.g. `/LAYER_.*_LAST/`) and
top-5 logits (e.g. `/top-5/`). For specific kernel checks, call
`window.testHarness.runMatmul(...)` or related helpers from the console.

**Manual browser testing:** Use the onboarding commands and canonical URLs in
[getting-started.md](getting-started.md), then use the test-mode templates in
[testing.md](testing.md).

---

## 12. Performance Debugging

### GPU Submit Tracking

DOPPLER includes submit tracking to measure per-token GPU overhead. The submit
tracker and command recorder below are repository modules
(`src/gpu/submit-tracker.js`, `src/gpu/command-recorder.js`), not package
exports; the paths are relative to `tests/harness.html` on a repo checkout.

```javascript
const { setTrackSubmits, resetSubmitStats, logSubmitStats } =
  await import('../src/gpu/submit-tracker.js');

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

```javascript
const { createCommandRecorder } = await import('../src/gpu/command-recorder.js');
const { recordMatmul } = await import('../src/gpu/kernels/matmul.js');
const { recordRMSNorm } = await import('../src/gpu/kernels/rmsnorm.js');

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
| `src/gpu/command-recorder.js` | Batched command recording |
| `src/gpu/submit-tracker.js` | GPU submit statistics |
| `src/inference/pipelines/text.js` | Forward pass orchestration |
| `src/inference/pipelines/text/execution/layer.js` | do* wrappers for run/record variants |

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

## For Debugging Agents

Follow the [Inference Debug Protocol](agents/debug-protocol.md) and the
[Debug Playbook](debug-playbook.md). Use this page for the Doppler-specific
tools: the [Quick Diagnosis Table](#quick-diagnosis-table), trace and pipeline
debug config ([Section 2](#2-toggleable-debug-categories)), and the
[Postmortem Index](#postmortem-index).

### Key Files for Debugging

| File | Debug Focus |
|------|-------------|
| `src/inference/pipelines/text.js` | Overall flow, token loop |
| `src/inference/pipelines/text/execution/layer.js` | Per-layer processing |
| `src/inference/pipelines/text/attention/` | KV cache, RoPE, attention |
| `src/inference/pipelines/text/ffn/` | FFN activation gating |
| `src/gpu/kernels/*.js` | Kernel selection and dispatch |
| `src/loader/doppler-loader.js` | Weight loading, dequantization |

---

*Last updated: 2026-11-26*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
Canonical reference:
[WGSL style guide: Kernel Compatibility and Overrides](style/wgsl-style-guide.md#kernel-compatibility-and-overrides).

Other docs link to this section; it points to the WGSL style guide, which covers:
- Runtime kernel paths (`runtime.inference.kernelPath` as an inline
  execution-v1 object or `null`; precedence is in
  [conversion-runtime-contract.md](conversion-runtime-contract.md)).
- RDRR layout vs runtime kernel variants and their GPU feature requirements.
- The OPFS purge helper (clearing stored manifests and shards when debugging).
