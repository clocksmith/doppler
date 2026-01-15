# Doppler Test Harness

## Overview

This directory contains the unified test harness for Doppler inference and GPU kernels.
The harness supports multiple modes via URL parameter, consolidating what were previously
separate test pages.

> **Note:** Unit tests (Vitest) live in [`tests/unit/`](../tests/unit/).
> GPU kernel correctness specs live in [`tests/kernels/`](../tests/kernels/).

## Unified Test Harness

**URL:** `http://localhost:8080/doppler/tests/harness.html`

### Modes

| Mode | URL | Purpose |
|------|-----|---------|
| `kernels` | `?mode=kernels` | GPU kernel correctness tests |
| `inference` | `?mode=inference` | Inference pipeline tests |
| `bench` | `?mode=bench` | Benchmark injection shell |

### Mode: Kernels

Tests 30+ kernel functions (matmul, attention, softmax, RoPE, etc.) by comparing
GPU output against CPU reference implementations.

**Features:**
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

### Mode: Inference

CI/automation testing of the inference pipeline.

**Features:**
- Model loading and token generation
- Query param automation for CI integration
- Playwright integration via `window.testState` and `window.pipeline`

**Query Parameters:**

| Param | Description | Example |
|-------|-------------|---------|
| `model` | Model ID | `&model=gemma3-1b-q4` |
| `prompt` | Prompt text | `&prompt=Hello%20world` |
| `autorun=1` | Auto-run on load | `&autorun=1` |
| `kernelPath` | Kernel path override | `&kernelPath="gemma2-q4k-fused-f16a"` |
| `debug=1` | Enable debug logging | `&debug=1` |
| `profile=1` | GPU timestamp profiling | `&profile=1` |
| `trace` | Trace level: quick or full | `&trace=quick` |
| `noChat` | Disable chat template | `&noChat` |
| `chat` | Force chat template | `&chat` |

**Playwright automation:**
```javascript
// Wait for test to complete
await page.waitForFunction(() => window.testState?.done === true);

// Check results
const state = await page.evaluate(() => window.testState);
console.log('Output:', state.output);
console.log('Errors:', state.errors);
```

### Mode: Bench

Shell for Playwright to inject benchmark scripts. Initializes WebGPU and signals
ready via `window.dopplerReady = true`.

Used by `npm run bench` — the CLI injects the `PipelineBenchmark` class.

---

## Running Tests

```bash
# Start dev server
npm start

# Run all kernel tests (default)
npm test

# Run quick kernel subset
npm test -- --quick

# Run inference smoke test
npm test -- --inference

# Manual browser testing
open "http://localhost:8080/doppler/tests/harness.html?mode=inference&model=gemma3-1b-q4&autorun=1"

# With runtime config
open "http://localhost:8080/doppler/tests/harness.html?mode=inference&model=gemma3-1b-q4&autorun=1&runtimeConfig={...}"
```

## Shared Test Utilities

The `inference/test-harness.js` module provides shared utilities:

```typescript
import {
  discoverModels,           // Fetch models from /api/models
  parseRuntimeOverridesFromURL, // Parse runtimeConfig from URL params
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
- [TESTING.md](../docs/TESTING.md) — Full testing documentation
