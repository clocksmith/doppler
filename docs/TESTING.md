# DOPPLER Testing Guide

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
| `docs/design/BENCHMARK_HARNESS.md` | Benchmark methodology |
| `docs/TEST_RESULTS.md` | Test session log |
| `tests/kernels/TODO.md` | Implementation status |
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

# Multiple runs for statistics
doppler bench --runs 3

# Custom prompt size (xs, short, medium, long)
doppler bench --prompt xs
```

**Prompt sizes:** `xs` (6-10 tokens), `short`, `medium`, `long`

### 5. Log Levels

Control loader output verbosity with CLI flags:

| CLI Flag | Level | Shows |
|----------|-------|-------|
| (default) | info | Phase starts/ends, totals |
| `--verbose` | verbose | + Per-shard source, per-layer timing |
| `--trace` | trace | + Tensor shapes, dequant ops |
| `--quiet` | silent | Errors only |

```bash
# Show shard sources (RAM/OPFS/network)
doppler bench --verbose

# Quiet mode for CI
doppler test --quiet
```

## Prerequisites

- **CLI commands:** Server auto-starts, no manual setup needed
- **For inference/pipeline tests:** Ensure model is available at `/doppler/models/<model-name>/`
- **For headed mode:** Chrome with WebGPU support required

## Test URLs (Manual Browser Testing)

For manual browser testing, start the server first: `npm start`

Open in browser while dev server is running:

- **Inference test page:** http://localhost:8080/doppler/tests/test-inference.html
  - Add `?model=gemma3-1b-q4` to specify model
  - Add `&autorun=1` to auto-start test
- **Kernel test page:** http://localhost:8080/doppler/tests/kernels/browser/index.html
- **Demo UI:** http://localhost:8080/d

## Adding New Tests

### Kernel Tests
Add to `tests/kernels/` and update `cli/index.js` switch statement.

### Inference Tests
Modify `tests/test-inference.html` for test logic.

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
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes, CLI flags (`--kernel-path`, `--kernel-profile`), and the OPFS purge helper.

