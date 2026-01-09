# DOPPLER Kernel Tests

GPU kernel correctness validation and benchmarks for the DOPPLER WebGPU inference engine.

## Quick Start

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

## Structure

```
tests/kernels/
├── harness/              # Test utilities (tolerance, buffer-utils, benchmark)
├── reference/            # CPU reference implementations
├── correctness/          # Kernel correctness tests (*.spec.js)
├── benchmarks/           # Performance benchmarks (*.bench.js)
└── browser/              # WebGPU test page
```

## Kernels Tested

| Kernel | Reference | Correctness | Benchmark |
|--------|-----------|-------------|-----------|
| matmul | matmul.js | matmul.spec.js | matmul.bench.js |
| softmax | softmax.js | softmax.spec.js | all-kernels.bench.js |
| attention | attention.js | attention.spec.js | all-kernels.bench.js |
| rmsnorm | rmsnorm.js | rmsnorm.spec.js | all-kernels.bench.js |
| rope | rope.js | rope.spec.js | all-kernels.bench.js |
| silu | silu.js | silu.spec.js | all-kernels.bench.js |
| gather | gather.js | gather.spec.js | all-kernels.bench.js |
| residual | residual.js | residual.spec.js | all-kernels.bench.js |
| topk | topk.js | topk.spec.js | moe-pipeline.bench.js |
| scatter-add | scatter-add.js | scatter-add.spec.js | moe-pipeline.bench.js |
| moe-gather | moe-gather.js | moe-gather.spec.js | moe-pipeline.bench.js |
| dequant | dequant.js | dequant.spec.js | all-kernels.bench.js |

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

## Test Configuration

Tests use Playwright with Chrome/Chromium and WebGPU flags:
- `--enable-unsafe-webgpu` - Enable WebGPU API
- `--enable-features=Vulkan` - Use Vulkan backend

For CI, use SwiftShader for software rendering.

## Related

- [BENCHMARKS.md](./BENCHMARKS.md) - Benchmark baselines and expected ranges
- [docs/TESTING.md](../docs/TESTING.md) - Full testing guide
- [docs/design/KERNEL_TESTING.md](../docs/design/KERNEL_TESTING.md) - Testing specification
