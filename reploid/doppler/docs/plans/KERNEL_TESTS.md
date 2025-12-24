# Kernel Tests Plan

**Part of:** [VISION.md](../VISION.md) - Foundation for all phases

Kernel correctness and microbenchmark tracking.

**Implementation:** See `kernel-tests/` directory for actual test code.

---

## Status: Complete

All kernel correctness tests and benchmarks implemented (Dec 2025).

---

## Test Coverage

### Reference Implementations (Complete)

| Kernel | File | Status |
|--------|------|--------|
| matmul | `src/reference/matmul.ts` | Done |
| softmax | `src/reference/softmax.ts` | Done |
| attention | `src/reference/attention.ts` | Done |
| rmsnorm | `src/reference/rmsnorm.ts` | Done |
| rope | `src/reference/rope.ts` | Done |
| silu | `src/reference/silu.ts` | Done |
| gather | `src/reference/gather.ts` | Done |
| residual | `src/reference/residual.ts` | Done |
| topk | `src/reference/topk.ts` | Done |
| scatter-add | `src/reference/scatter-add.ts` | Done |
| moe-gather | `src/reference/moe-gather.ts` | Done |
| dequant | `src/reference/dequant.ts` | Done |

### Correctness Tests (Complete)

| Kernel | File | Status |
|--------|------|--------|
| matmul | `tests/correctness/matmul.spec.ts` | Done |
| softmax | `tests/correctness/softmax.spec.ts` | Done |
| attention | `tests/correctness/attention.spec.ts` | Done |
| rmsnorm | `tests/correctness/rmsnorm.spec.ts` | Done |
| rope | `tests/correctness/rope.spec.ts` | Done |
| silu | `tests/correctness/silu.spec.ts` | Done |
| gather | `tests/correctness/gather.spec.ts` | Done |
| residual | `tests/correctness/residual.spec.ts` | Done |
| topk | `tests/correctness/topk.spec.ts` | Done |
| scatter-add | `tests/correctness/scatter-add.spec.ts` | Done |
| moe-gather | `tests/correctness/moe-gather.spec.ts` | Done |
| dequant | `tests/correctness/dequant.spec.ts` | Done |

### Benchmarks (Complete)

| Benchmark | File | Status |
|-----------|------|--------|
| matmul | `tests/benchmarks/matmul.bench.ts` | Done |
| MoE pipeline | `tests/benchmarks/moe-pipeline.bench.ts` | Done |
| All kernels | `tests/benchmarks/all-kernels.bench.ts` | Done |

---

## Tolerances

| Kernel | Tolerance | Notes |
|--------|-----------|-------|
| matmul_f32 | rtol=1e-5 | Standard FP32 |
| matmul_f16 | rtol=1e-2 | FP16 has ~3 decimal digits |
| softmax | rtol=1e-5 | Numerically stable |
| topk indices | exact | Must match exactly |
| topk weights | rtol=1e-5 | After renormalization |
| scatter_add | rtol=1e-5 | Weighted sum |
| rmsnorm | rtol=1e-4 | Reduction tolerance |
| attention | rtol=1e-3 | Multiple reductions |
| dequant | rtol=1e-4 | Quantization error |

---

## Running Tests

```bash
cd kernel-tests
npm install

# Run all tests
npm test

# Run GPU tests only
npm run test:gpu

# Run GPU tests headed (see browser)
npm run test:gpu:headed

# Run benchmarks
npm run bench

# Serve test page for manual testing
npm run serve
```

---

## Remaining Work

| Task | Priority | Status |
|------|----------|--------|
| GitHub Actions CI | P2 | TODO |
| CI report script | P2 | TODO |
| Regression detection | P2 | TODO |

---

## Related

- [BENCHMARKS.md](../../kernel-tests/BENCHMARKS.md) - Baseline expectations
- [BENCHMARK_HARNESS.md](../spec/BENCHMARK_HARNESS.md) - Methodology spec
- [KERNEL_TESTING.md](../spec/KERNEL_TESTING.md) - Test design spec

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

