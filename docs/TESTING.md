# DOPPLER Testing

## Testing Strategy

## Test Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DOPPLER Test Stack                      │
├─────────────────────────────────────────────────────────────┤
│  End-to-end inference + bench (browser harness + demo)      │
├─────────────────────────────────────────────────────────────┤
│  GPU kernel correctness (tests/kernels/browser)             │
├─────────────────────────────────────────────────────────────┤
│  Training kernels + parity (tests/training/browser)         │
└─────────────────────────────────────────────────────────────┘
```

**Key Points:**
- **All test flows run in the browser** (no Node/Vitest/Playwright).
- **Runtime config drives behavior**; set `runtime.shared.tooling.intent` for verify/debug/bench.
- **Models load from OPFS/IndexedDB or `/models/<id>` when served locally.**

## Related Documentation

| Doc | Purpose |
|-----|---------|
| [architecture.md](architecture.md) | System architecture context |
| `../tests/kernels/README.md` | Kernel test coverage and design notes |
| `style/benchmark-style-guide.md` | Benchmark methodology |
| Internal postmortems (private wrapper repo) | Test-related incident history |
| `../tests/kernels/benchmarks.md` | Benchmark baselines |

---

## Quick Reference

| Action | How | When to Use |
|--------|-----|-------------|
| Kernel correctness | `tests/harness.html` mode `kernels` | After GPU kernel changes |
| Inference smoke test | `tests/harness.html` mode `inference` | After pipeline changes |
| Benchmarks | Demo diagnostics (suite: `bench`) | Performance measurement |
| Training kernels | `tests/harness.html` mode `training` | Training work |

## Test Systems

### 1. Kernel Tests (mode: `kernels`)

GPU kernel correctness validation via the browser harness.

Example runtime config:

```json
{
  "shared": {
    "tooling": { "intent": "verify" },
    "harness": {
      "mode": "kernels",
      "autorun": true
    }
  }
}
```

Open:
`http://localhost:8080/tests/harness.html?runtimeConfig=...`

### 2. Inference Test (mode: `inference`)

End-to-end model loading and token generation.

Example runtime config:

```json
{
  "shared": {
    "tooling": { "intent": "verify" },
    "harness": {
      "mode": "inference",
      "autorun": true,
      "skipLoad": false,
      "modelId": "gemma3-1b-q4"
    }
  },
  "inference": {
    "prompt": "Hello from Doppler."
  }
}
```

### 3. Performance Benchmarks

Use the demo diagnostics UI (`/demo/`) and choose `bench` as the suite. The
bench runner uses `runtime.shared.benchmark.run` for warmups and timing.

### 4. Log Levels

Control loader output verbosity via runtime config:

| Config | Level | Shows |
|--------|-------|-------|
| `runtime.shared.debug.logLevel.defaultLogLevel=info` | info | Phase starts/ends, totals |
| `...=verbose` | verbose | + Per-shard source, per-layer timing |
| `runtime.shared.debug.trace.enabled=true` | trace | + Tensor shapes, dequant ops |
| `...=silent` | silent | Errors only |

## Prerequisites

- WebGPU-capable browser (Chrome/Chromium recommended)
- Static server for local files (e.g. `python3 -m http.server 8080`)

## Test URLs (Manual Browser Testing)

Start the server first: `python3 -m http.server 8080`

- **Unified test harness:** http://localhost:8080/tests/harness.html
  - Modes are configured via `runtime.shared.harness` and passed in `runtimeConfig`
  - The harness does not accept per-field URL overrides
- **Demo UI:** http://localhost:8080/demo/

Example (inference mode):
```
const cfg = {
  shared: { harness: { mode: 'inference', autorun: true, skipLoad: false, modelId: 'gemma3-1b-q4' } }
};
encodeURIComponent(JSON.stringify(cfg));
```

## Adding New Tests

### Kernel Tests
Add to `tests/kernels/browser/test-page.js` and reference the new kernel in the harness.

### Inference Tests
Modify `tests/harness.html` inference mode or `src/inference/browser-harness.js`.

### Benchmarks
Extend `tests/benchmarks/` and wire into the diagnostics flow as needed. The
browser harness lives in `src/inference/browser-harness.js`.

### Training Tests
Add to `tests/training/browser/test-page.js`.

#### Training Bench: EBM Recorded

The training harness includes an EBM viability benchmark (`ebm-recorded-bench`).
Its tunables are runtime-config driven:

- Dims: `runtime.shared.harness.trainingBench.ebmRecorded.dims` (`M`, `K`, `H`, `O`)
- Runs: `runtime.shared.benchmark.run.warmupRuns` + `runtime.shared.benchmark.run.timedRuns`

## CI Integration

Browser automation is not wired in this repo yet. Run the harness and diagnostics
flows locally for validation until a browser CI runner is added.

## Kernel Overrides & Compatibility
See `OPERATIONS.md#kernel-overrides--compatibility` (canonical section) and `style/wgsl-style-guide.md`.


## Kernel Testing

Defines the testing framework design for WGSL kernels and kernel combinations.

**Implementation status:** See `../tests/kernels/README.md`
**Benchmark baselines:** See `../tests/kernels/benchmarks.md`

---

## Goals

- Catch kernel correctness regressions early (especially after performance refactors).
- Provide unit tests for individual WGSL kernels.
- Provide integration tests for kernel sequences that occur in real inference.
- Make results interpretable across different GPUs and browsers.

---

## Core Concepts

### Kernel Unit Test

A kernel unit test runs a single WGSL entry point with known inputs and compares outputs to a reference implementation.

### Pipeline Segment Test

A segment test runs a small sequence of kernels that reflects a real subgraph, for example:

- RMSNorm -> QKV matmul -> RoPE -> attention -> residual add
- FFN dense: matmul -> activation -> matmul -> residual add
- MoE: router logits -> softmax+topk -> expert compute -> scatter-add

### Reference Implementation

Use a CPU reference in JavaScript for correctness checks. The reference must:

- Match the math of the WGSL kernel
- Be deterministic
- Clearly define tolerances for floating-point differences

---

## What To Test (Required Coverage)

### Kernels

- Dequantization: Q4_K and MXFP4 paths
- Matmul: f32, f16, mixed precision variants, and the M=1 decode fast path
- RMSNorm
- RoPE
- Attention (all tiers and f16 KV variants)
- Softmax
- Activations: SiLU, GeLU, SwiGLU
- Gather (embedding)
- BiasAdd and ResidualAdd
- TopK and SoftmaxTopK
- MoE kernels: gather and scatter-add

---

## Test Data Strategy

### Deterministic Inputs

Use fixed seeds and explicit arrays rather than random runtime generation.

Recommended patterns:

- Small shapes that fit in one workgroup, and medium shapes that span multiple workgroups.
- Edge shapes: headDim boundaries (64, 128, 256), kvHeads boundaries, and vocab sizes (small mock vocab).
- Numeric edge cases: zeros, large magnitudes, near-underflow values.

### Tolerances

Report tolerances per kernel:

- f32 outputs: default `atol = 1e-4`, `rtol = 1e-4`
- f16 outputs: default `atol = 5e-3`, `rtol = 5e-3`

Attention and softmax tests should check:

- Probability mass sums to approximately 1
- Masking correctness (causal)
- Stability on long sequences (no NaNs, no Infs)

---

## Pipeline Combination Tests (Required)

Define a small set of "golden" segment tests that combine kernels in the same way inference does.

Examples:

1. Dense layer mini-forward (no MoE)
   - Embedding -> RMSNorm -> QKV -> RoPE -> Attention -> Residual -> FFN -> Residual

2. Decode attention step
   - Single-token Q -> cached KV -> attention decode kernel -> residual

3. MoE routing step
   - Router matmul -> softmax+topk -> scatter-add combine

Each segment test should:

- Use small tensor sizes
- Use fixed parameters
- Compare a final output tensor to a CPU reference

---

## Correctness Oracles (Recommended)

When a full CPU reference is expensive:

- Use invariant checks:
  - softmax sum close to 1
  - attention output bounded and finite
  - topk indices sorted by score
- Use cross-implementation checks:
  - compare two GPU variants (streaming attention vs tiled attention) on the same input and confirm close outputs

---

## Debugging Aids (Recommended)

For failures, store:

- WGSL variant name and entry point
- Device capabilities (`shader-f16`, `subgroups`, limits)
- Max absolute error and index of the worst element
- A small excerpt of inputs and outputs

---

## Implementation Layout

| Type | Location | Notes |
|------|----------|-------|
| Kernel tests | `tests/kernels/browser/test-page.js` | Harness-based kernel checks |
| Benchmarks | `tests/benchmarks/` | Performance measurement |
| References | `tests/kernels/reference/` | CPU reference implementations |
| Harness | `tests/kernels/harness/` | Test utilities |
| Browser | `tests/kernels/browser/` | WebGPU test page |

---

*Last updated: December 2025*

## Kernel Overrides & Compatibility
See `OPERATIONS.md#kernel-overrides--compatibility` (canonical section) and `style/wgsl-style-guide.md`.
