# Doppler Test Harness

## Overview

This directory contains the browser-only test harnesses for Doppler inference and GPU kernels.
The unified harness supports multiple modes via runtime config.

## Unified Test Harness

**URL:** `http://localhost:8080/tests/harness.html` (serve repo root with a static server)

### Modes

Modes are configured in `runtime.shared.harness` and passed via the `runtimeConfig`
URL parameter. The harness does not accept per-field query overrides.

| Mode | Purpose |
|------|---------|
| `kernels` | GPU kernel correctness tests |
| `inference` | Inference pipeline tests |
| `bench` | Benchmark runner shell |
| `training` | Training kernel tests |

### Mode: Kernels

Tests 30+ kernel functions (matmul, attention, softmax, RoPE, etc.) by comparing
GPU output against CPU reference implementations.

**Features:**
- `window.testHarness` for manual runs
- GPU capability detection (F16, subgroups, memory limits)

```javascript
// Run a specific kernel test
const result = await window.testHarness.runMatmul({ M: 128, N: 256, K: 64 });
```

### Mode: Inference

Manual testing of the inference pipeline.

**Features:**
- Model loading and token generation
- Query param automation via `runtimeConfig`
- Status exposed on `window.testState` and `window.pipeline`

**Query Parameters:**

| Param | Description | Example |
|-------|-------------|---------|
| `runtimeConfig` | JSON-encoded runtime config | `&runtimeConfig={...}` |
| `configChain` | JSON-encoded config chain | `&configChain=["debug","default"]` |

### Mode: Bench

Initializes WebGPU and signals ready via `window.dopplerReady = true`.
You can run `runBrowserSuite({ suite: 'bench', modelId })` from the console or use the demo diagnostics UI.

---

## Running Tests

```bash
# Start a static server
python3 -m http.server 8080

# Manual browser testing (runtimeConfig defines harness mode)
# Example runtimeConfig:
# {"shared":{"harness":{"mode":"inference","autorun":true,"skipLoad":false,"modelId":"gemma3-1b-q4"}}}
# Encode with encodeURIComponent(JSON.stringify(...)) in devtools.
```

## Shared Test Utilities

The `inference/test-harness.js` module provides shared utilities:

```typescript
import {
  discoverModels,             // Fetch models from /api/models (optional)
  parseRuntimeOverridesFromURL, // Parse runtimeConfig from URL params
  createHttpShardLoader,      // Create HTTP-based shard loader
  fetchManifest,              // Fetch and parse manifest.json
  initializeDevice,           // Initialize WebGPU device
  createTestState,            // Create standard test state object
} from '/src/inference/test-harness.js';
```

## Related Documentation

- [architecture.md](../docs/architecture.md) — System overview
- [KERNEL_COMPATIBILITY.md](../docs/KERNEL_COMPATIBILITY.md) — Kernel modes and runtime flags
- [RDRR_FORMAT.md](../docs/spec/RDRR_FORMAT.md) — Model format specification
- [testing.md](../docs/testing.md) — Full testing documentation
