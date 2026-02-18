# DOPPLER Kernel Tests

GPU kernel correctness validation and benchmarks for the DOPPLER WebGPU inference engine.

## Quick Start

```bash
# From repo root
python3 -m http.server 8080
```

Open the harness in a WebGPU-capable browser:
`http://localhost:8080/tests/harness.html?runtimeConfig=...`

## Structure

```
tests/kernels/
├── harness/              # Test utilities (tolerance, buffer-utils, benchmark)
├── reference/            # CPU reference implementations
└── browser/              # WebGPU test page
```

## Kernels Tested

| Kernel | Reference |
|--------|-----------|
| matmul | matmul.js |
| softmax | softmax.js |
| attention | attention.js |
| rmsnorm | rmsnorm.js |
| rope | rope.js |
| silu | silu.js |
| gather | gather.js |
| residual | residual.js |
| topk | topk.js |
| scatter-add | scatter-add.js |
| moe-gather | moe-gather.js |
| dequant | dequant.js |

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

Use Chrome/Chromium (or another WebGPU-capable browser). Ensure WebGPU is enabled
and that the browser allows file/HTTP access for local hosting.

## Related

- [benchmarks.md](./benchmarks.md) - Benchmark baselines and expected ranges
- [testing.md](../../docs/testing.md) - Full testing guide
- [operations.md](../../docs/operations.md) - Troubleshooting and validation workflows
