# DOPPLER Testing

## Testing Strategy

> For cross-project test strategy and Ouroboros testing, see [TEST_PLAN.md](../../TEST_PLAN.md)

## Test Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DOPPLER Test Pyramid                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              End-to-End Inference                    │   │
│  │         doppler test inference (Playwright)          │   │
│  │    Model load → Pipeline → Generate → Validate       │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              GPU Kernel Correctness                  │   │
│  │         doppler test kernels (Playwright+WebGPU)     │   │
│  │    WGSL kernels vs CPU reference implementations     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 CPU Unit Tests                       │   │
│  │              npm run test:vitest                     │   │
│  │    Tokenizer, manifest parsing, utilities            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Points:**
- **GPU tests** require Chrome/Chromium with WebGPU (Playwright handles this)
- **CPU tests** run in Node.js via Vitest (no GPU needed)
- **Benchmarks** measure performance, not correctness (separate from tests)

## Related Documentation

| Doc | Purpose |
|-----|---------|
| [TEST_PLAN.md](../../TEST_PLAN.md) | Cross-project test strategy |
| [ARCHITECTURE.md](../../ARCHITECTURE.md) | Category-2t architecture |
| `docs/design/KERNEL_TESTING.md` | Testing specification/design |
| `docs/style/BENCHMARK_STYLE_GUIDE.md` | Benchmark methodology |
| `docs/TEST_RESULTS.md` | Test session log |
| `tests/kernels/README.md` | Kernel test coverage |
| `tests/kernels/BENCHMARKS.md` | Benchmark baselines |

---

## Quick Reference

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `doppler test` | Quick kernel tests | CI, before commits |
| `doppler test --full` | Full kernel correctness | After GPU kernel changes |
| `doppler test --inference` | Model load + generate | After pipeline changes |
| `npm run test:vitest` | CPU unit tests | After non-GPU code changes |
| `doppler bench` | Inference benchmark | Performance measurement |
| `doppler bench --kernels` | Kernel benchmarks | Before/after kernel optimizations |
| `doppler debug` | Debug with trace | Investigating inference bugs |

## Test Systems

### 1. Kernel Tests (`doppler test kernels`)

GPU kernel correctness validation via Playwright + WebGPU.

```bash
# Quick validation (CI default)
doppler test quick

# Full kernel suite
doppler test kernels

# With visible browser
doppler test kernels --headed

# Filter specific kernel
doppler test kernels --filter matmul

# Save results to file
doppler test kernels -o results.json

# With performance benchmarks
doppler test kernels --perf
```

**Kernels tested:** matmul, attention, rmsnorm, softmax, rope, silu, gather, scatter-add, moe-gather, residual, topk, dequant

### 2. Inference Test (`doppler test inference`)

End-to-end model loading and token generation.

```bash
# Default model (gemma3-1b-q4)
doppler test inference

# Specific model
doppler test inference --model mistral-7b-q4

# With visible browser
doppler test inference --headed
```

**What it tests:**
- WebGPU initialization
- Model manifest parsing
- Shard loading
- Pipeline creation
- Token generation (50 tokens)

### 3. CPU Unit Tests (`npm run test:vitest`)

Non-GPU JavaScript/TypeScript unit tests.

```bash
npm run test:vitest           # Run once
npm run test:vitest:watch     # Watch mode
npm run test:vitest:ui        # Interactive UI
npm run test:vitest:coverage  # With coverage report
```

### 4. Performance Benchmarks (`doppler bench`)

Use `doppler bench` for performance measurement.

```bash
# Full inference benchmark
doppler bench

# Kernel microbenchmarks
doppler bench --kernels

# Multiple runs for statistics (via config)
doppler bench --config bench
```

**Prompt sizes:** `xs` (6-10 tokens), `short`, `medium`, `long` (set via `runtime.shared.benchmark.run.promptName`)

### 5. Log Levels

Control loader output verbosity via runtime config:

| Config | Level | Shows |
|--------|-------|-------|
| `runtime.shared.debug.logLevel.defaultLogLevel=info` | info | Phase starts/ends, totals |
| `...=verbose` | verbose | + Per-shard source, per-layer timing |
| `runtime.shared.debug.trace.enabled=true` | trace | + Tensor shapes, dequant ops |
| `...=silent` | silent | Errors only |

```bash
# Show shard sources (RAM/OPFS/network)
doppler bench --config debug
```

## Prerequisites

- **CLI commands:** Server auto-starts, no manual setup needed
- **For inference/pipeline tests:** Ensure model is available at `/doppler/models/<model-name>/`
- **For headed mode:** Chrome with WebGPU support required

## Test URLs (Manual Browser Testing)

For manual browser testing, start the server first: `npm start`

Open in browser while dev server is running:

- **Unified test harness:** http://localhost:8080/doppler/tests/harness.html
  - Modes are configured via `runtime.shared.harness` and passed in `runtimeConfig`
  - The harness does not accept per-field URL overrides
- **Demo UI:** http://localhost:8080/d

Example (inference mode):
```bash
node -e "const cfg={shared:{harness:{mode:'inference',autorun:true,skipLoad:false,modelId:'gemma3-1b-q4'}}};console.log(encodeURIComponent(JSON.stringify(cfg)));"
# Paste output into:
# http://localhost:8080/doppler/tests/harness.html?runtimeConfig=...
```

## Adding New Tests

### Kernel Tests
Add to `tests/kernels/` and update `cli/index.js` switch statement.

### Inference Tests
Modify `tests/harness.html` inference mode for test logic.

### Unit Tests
Add `.test.js` files to `tests/` directory.

## CI Integration

GitHub Actions runs `npm test` (quick kernel suite) on push/PR.

For local CI simulation:
```bash
doppler test quick && npm run test:vitest
```

For full validation before merging:
```bash
doppler test kernels && doppler test inference && npm run test:vitest
```

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/style/WGSL_STYLE_GUIDE.md` for runtime kernel modes and the OPFS purge helper.


## Kernel Testing

Defines the testing framework design for WGSL kernels and kernel combinations.

**Implementation status:** See `tests/kernels/README.md`
**Benchmark baselines:** See `tests/kernels/BENCHMARKS.md`

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
| Kernel tests | `tests/kernels/tests/correctness/` | Unit tests per kernel |
| Benchmarks | `tests/kernels/tests/benchmarks/` | Performance measurement |
| References | `tests/kernels/src/reference/` | CPU reference implementations |
| Harness | `tests/kernels/src/harness/` | Test utilities |
| Browser | `tests/kernels/browser/` | WebGPU test page |

For pipeline segment tests, use `tests/segments/` with CPU refs and saved JSON outputs.

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/style/WGSL_STYLE_GUIDE.md` for runtime kernel modes and the OPFS purge helper.


## Known-Good Matrix

This matrix defines the smallest set of fixtures that must load and produce
stable outputs. Keep it narrow and deterministic.

## Fixtures

| Fixture | Format | Quant | Notes | Test |
| --- | --- | --- | --- | --- |
| `tests/fixtures/mini-model` | RDRR | F32 | Small 2-layer transformer, bundled tokenizer | `tests/correctness/known-good-fixtures.spec.js` |
| `tests/fixtures/tiny-model` | RDRR | F32 | Alternate shape/layout for loader coverage | `tests/correctness/known-good-fixtures.spec.js` |
| `tests/fixtures/sample.gguf` | GGUF | F32 | Parser coverage only (no inference) | `tests/unit/formats-gguf.test.js` |

## Output Checksums

Known-good outputs are stored in:

```
tests/fixtures/known-good-outputs.json
```

To update:

```
DOPPLER_UPDATE_KNOWN_GOOD=1 npx playwright test -c tests/correctness/playwright.config.js
```
