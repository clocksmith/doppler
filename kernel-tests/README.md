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
kernel-tests/
├── src/
│   ├── harness/          # Test utilities (tolerance, buffer-utils, benchmark)
│   └── reference/        # CPU reference implementations
├── tests/
│   ├── correctness/      # Kernel correctness tests (*.spec.ts)
│   └── benchmarks/       # Performance benchmarks (*.bench.ts)
└── browser/              # WebGPU test page
```

## Kernels Tested

| Kernel | Reference | Correctness | Benchmark |
|--------|-----------|-------------|-----------|
| matmul | matmul.ts | matmul.spec.ts | matmul.bench.ts |
| softmax | softmax.ts | softmax.spec.ts | all-kernels.bench.ts |
| attention | attention.ts | attention.spec.ts | all-kernels.bench.ts |
| rmsnorm | rmsnorm.ts | rmsnorm.spec.ts | all-kernels.bench.ts |
| rope | rope.ts | rope.spec.ts | all-kernels.bench.ts |
| silu | silu.ts | silu.spec.ts | all-kernels.bench.ts |
| gather | gather.ts | gather.spec.ts | all-kernels.bench.ts |
| residual | residual.ts | residual.spec.ts | all-kernels.bench.ts |
| topk | topk.ts | topk.spec.ts | moe-pipeline.bench.ts |
| scatter-add | scatter-add.ts | scatter-add.spec.ts | moe-pipeline.bench.ts |
| moe-gather | moe-gather.ts | moe-gather.spec.ts | moe-pipeline.bench.ts |
| dequant | dequant.ts | dequant.spec.ts | all-kernels.bench.ts |

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
