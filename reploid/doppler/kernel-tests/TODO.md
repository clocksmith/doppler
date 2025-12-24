# Kernel Validation TODO

> Status: Complete | Started: Dec 2025

## Phase 1: Workspace Setup ✅

- [x] `package.json` - workspace config with vitest + playwright
- [x] `vitest.config.ts` - unit test config
- [x] `playwright.config.ts` - GPU test config with WebGPU flags
- [x] `browser/index.html` - test runner page
- [x] `browser/test-page.ts` - GPU test initialization

## Phase 2: Test Harness ✅

- [x] `src/harness/tolerance.ts` - FP comparison utilities
- [x] `src/harness/buffer-utils.ts` - GPU buffer helpers
- [x] `src/harness/benchmark.ts` - timing harness
- [x] `src/harness/index.ts` - exports

## Phase 3: Reference Implementations ✅

Priority (newly implemented MoE kernels):
- [x] `src/reference/topk.ts` - top-k selection
- [x] `src/reference/scatter-add.ts` - weighted scatter-add

Core kernels:
- [x] `src/reference/matmul.ts` - matrix multiplication
- [x] `src/reference/softmax.ts` - online softmax
- [x] `src/reference/attention.ts` - multi-head attention

Remaining:
- [x] `src/reference/rmsnorm.ts`
- [x] `src/reference/rope.ts`
- [x] `src/reference/silu.ts`
- [x] `src/reference/gather.ts`
- [x] `src/reference/residual.ts`
- [x] `src/reference/moe-gather.ts`
- [x] `src/reference/dequant.ts`
- [x] `src/reference/index.ts` - exports

## Phase 4: Correctness Tests ✅

Priority:
- [x] `tests/correctness/setup.ts` - test utilities
- [x] `tests/correctness/topk.spec.ts`
- [x] `tests/correctness/scatter-add.spec.ts`
- [x] `tests/correctness/matmul.spec.ts`

Remaining:
- [x] `tests/correctness/softmax.spec.ts`
- [x] `tests/correctness/attention.spec.ts`
- [x] `tests/correctness/rmsnorm.spec.ts`
- [x] `tests/correctness/rope.spec.ts`
- [x] `tests/correctness/silu.spec.ts`
- [x] `tests/correctness/gather.spec.ts`
- [x] `tests/correctness/residual.spec.ts`
- [x] `tests/correctness/moe-gather.spec.ts`
- [x] `tests/correctness/dequant.spec.ts`

## Phase 5: Benchmarks ✅

- [x] `tests/benchmarks/config.ts` - workload configs (Llama/Mixtral sizes)
- [x] `tests/benchmarks/matmul.bench.ts`
- [x] `tests/benchmarks/moe-pipeline.bench.ts` - end-to-end MoE routing
- [x] `tests/benchmarks/all-kernels.bench.ts` - summary benchmark

## Phase 6: Integration ✅

- [x] Wire up `browser/test-page.ts` to actual GPU kernels
- [x] Expose `window.testHarness` for Playwright tests
- [x] Create `BENCHMARKS.md` with baseline documentation

## Phase 7: CI (Optional)

- [ ] GitHub Actions workflow
- [ ] CI report script

---

## Running Tests

```bash
# Install dependencies
cd kernel-tests
npm install

# Run all tests (unit + GPU)
npm test

# Run GPU tests only (requires Chrome with WebGPU)
npm run test:gpu

# Run GPU tests in headed mode (see browser)
npm run test:gpu:headed

# Run benchmarks
npm run bench

# Serve test page for manual testing
npm run serve
# Then open http://localhost:8080 in Chrome
```

## Test Configuration

The tests use Playwright with Chrome/Chromium with WebGPU flags:
- `--enable-unsafe-webgpu` - Enable WebGPU API
- `--enable-features=Vulkan` - Use Vulkan backend (better performance)

For CI, consider using SwiftShader for software rendering.

## Kernel Tolerances

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

## Progress Log

### Dec 2025
- Created workspace structure
- Completed Phase 1: Workspace setup (package.json, vitest, playwright configs)
- Completed Phase 2: Test harness (tolerance, buffer-utils, benchmark)
- Completed Phase 3: All 12 reference implementations
- Completed Phase 4: All 12 correctness test files
- Completed Phase 5: Benchmark suite with model configs
- Completed Phase 6: Test harness integration with GPU kernels

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

