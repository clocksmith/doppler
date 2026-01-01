# Doppler Testing Pages

## Overview

This directory contains manual test harnesses for Doppler inference and GPU kernels.
These pages are distinct from unit tests (Vitest) and Playwright correctness tests.

## Test Pages

### test-inference.html — Inference Harness

**Purpose:** CI/automation testing of the inference pipeline
**URL:** `http://localhost:8080/doppler/tests/test-inference.html`

**Features:**
- Model selection dropdown (auto-populated from `/api/models`)
- Single inference test with token streaming
- Batch compare test (batchSize=1 vs batchSize=N)
- Query param automation for CI integration
- Playwright integration via `window.testState`

**Query Parameters:**

| Param | Description | Example |
|-------|-------------|---------|
| `model` | Pre-select model ID | `?model=gemma3-1b-q4` |
| `prompt` | Pre-fill prompt text | `?prompt=Hello%20world` |
| `autorun=1` | Auto-run test on page load | `?autorun=1` |
| `kernelHints` | JSON kernel hints override | `?kernelHints={"q4kMatmul":"fused_q4k"}` |
| `attentionKernel` | Attention kernel override | `?attentionKernel=tiled_large` |
| `computePrecision` | Compute precision (f16/f32/auto) | `?computePrecision=f16` |
| `q4kMatmul` | Q4K matmul strategy | `?q4kMatmul=dequant_f16` |
| `debug=1` | Enable debug logging | `?debug=1` |
| `profile=1` | Enable GPU timestamp profiling | `?profile=1` |
| `trace` | Trace level: quick or full | `?trace=quick` |
| `debugLayers` | Comma-separated layer indices | `?debugLayers=0,5,10` |

**Kernel Trace Visualization:**
When `trace`, `debug`, or `profile` params are set, the page displays a kernel execution trace at the top showing:
- Which kernels executed and in what order
- Timing for each operation
- Indentation to show sub-operations (attention, FFN)

**Playwright automation:**
```javascript
// Wait for test to complete
await page.waitForFunction(() => window.testState?.done === true);

// Check results
const state = await page.evaluate(() => window.testState);
console.log('Output:', state.output);
console.log('Errors:', state.errors);
```

---

### ../tools/test-kernel-selection.html — Kernel Selection Debug

**Purpose:** Verify manifest kernel hints are loaded correctly
**URL:** `http://localhost:8080/doppler/tools/test-kernel-selection.html`

**When to use:**
- Debugging kernel hint configuration in manifest.json
- Verifying Q4K layout detection
- Checking hint priority resolution (manifest → profile → runtime)

**Output:** Console logs showing kernel selection decisions.

---

### ../kernel-tests/browser/index.html — GPU Kernel Test Runner

**Purpose:** Validate individual GPU kernel correctness
**URL:** `http://localhost:8080/doppler/kernel-tests/browser/`

**Features:**
- Tests 30+ kernel functions (matmul, attention, softmax, RoPE, etc.)
- Compares GPU output against CPU reference implementations
- `window.testHarness` for Playwright automation
- GPU capability detection (F16, subgroups, memory limits)

**Playwright automation:**
```javascript
// Run a specific kernel test
const result = await page.evaluate(async () => {
  const { testHarness } = window;
  return await testHarness.runMatmul({ M: 128, N: 256, K: 64 });
});
```

---

## Running Tests

```bash
# Start dev server
doppler serve

# Run Playwright kernel tests
cd kernel-tests && npm test

# Run inference smoke test
open "http://localhost:8080/doppler/tests/test-inference.html?model=gemma3-1b-q4&autorun=1"

# Run with specific kernel hints
open "http://localhost:8080/doppler/tests/test-inference.html?model=gemma3-1b-q4&autorun=1&q4kMatmul=dequant_f16"
```

## Shared Test Utilities

The `inference/test-harness.ts` module provides shared utilities:

```typescript
import {
  discoverModels,           // Fetch models from /api/models
  parseRuntimeOverridesFromURL, // Parse kernel hints from URL params
  createHttpShardLoader,    // Create HTTP-based shard loader
  fetchManifest,            // Fetch and parse manifest.json
  initializeDevice,         // Initialize WebGPU device
  createTestState,          // Create standard test state object
} from '/doppler/dist/inference/test-harness.js';
```

## Related Documentation

- [ARCHITECTURE.md](../docs/ARCHITECTURE.md) — System overview
- [KERNEL_COMPATIBILITY.md](../docs/KERNEL_COMPATIBILITY.md) — Kernel modes and runtime flags
- [RDRR_FORMAT.md](../docs/spec/RDRR_FORMAT.md) — Model format specification
