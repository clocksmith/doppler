# GPU Kernel Benchmark Baselines

This document describes the benchmark configurations and expected baseline performance for the Doppler GPU kernels.

## Test Configurations

### Model Sizes

| Config | Hidden | Intermediate | Heads | KV Heads | Head Dim | Experts | Top-K |
|--------|--------|--------------|-------|----------|----------|---------|-------|
| Small | 512 | 1408 | 8 | 8 | 64 | 4 | 2 |
| 7B | 4096 | 11008 | 32 | 32 | 128 | 8 | 2 |
| Mixtral | 4096 | 14336 | 32 | 8 | 128 | 8 | 2 |
| 13B | 5120 | 13824 | 40 | 40 | 128 | 8 | 2 |

### Sequence Lengths

| Name | Length | Use Case |
|------|--------|----------|
| Single | 1 | Decode (autoregressive) |
| Short | 32 | Short prompt |
| Medium | 128 | Typical prompt |
| Long | 512 | Long context |
| Very Long | 2048 | Extended context |

## Benchmark Results

> **Note**: Run benchmarks in the browser harness to collect numbers on your hardware.
> Results vary significantly by GPU vendor and driver version.

### Expected Ranges (Apple M1/M2/M3)

#### Matmul (GEMM)

| Configuration | Expected Time | Notes |
|--------------|---------------|-------|
| Matvec 1x4096x4096 | 0.5-2ms | Single token decode |
| 32x4096x4096 | 2-5ms | Short prompt |
| 128x4096x4096 | 5-15ms | Medium prompt |
| 4096x4096x4096 | 50-150ms | Large square |

#### Top-K Selection (MoE Routing)

| Tokens | Experts | Top-K | Expected Time |
|--------|---------|-------|---------------|
| 1 | 8 | 2 | <0.5ms |
| 32 | 8 | 2 | <1ms |
| 128 | 8 | 2 | 1-3ms |
| 128 | 16 | 4 | 2-5ms |

#### Scatter-Add (MoE Combine)

| Tokens | Hidden | Experts | Expected Time |
|--------|--------|---------|---------------|
| 1 | 4096 | 8 | <0.5ms |
| 32 | 4096 | 8 | 1-2ms |
| 128 | 4096 | 8 | 2-5ms |

#### MoE Full Pipeline (TopK + Scatter)

| Configuration | Expected Time | Tokens/sec |
|--------------|---------------|------------|
| Decode (1 token) | 1-3ms | 300-1000 |
| Short (32 tokens) | 5-10ms | 3000-6000 |
| Medium (128 tokens) | 10-25ms | 5000-12000 |

### Performance Guidelines

1. **Memory Bandwidth Bound**: Most kernels are memory-bound, not compute-bound
2. **Batch Size**: Larger batches amortize kernel launch overhead
3. **FP16 vs FP32**: FP16 can be 2x faster with half memory bandwidth
4. **Workgroup Size**: Optimal sizes vary by hardware (64-256 typical)

## Running Benchmarks

```bash
python3 -m http.server 8080
# Open http://localhost:8080/tests/harness.html?runtimeConfig=...
# Then invoke benchmark helpers from the console (window.testHarness).
```

For extra output, use `runtime.shared.debug.logLevel` or trace categories in the runtime config.

## Interpreting Results

### Key Metrics

- **Median Time**: Most stable metric, use for comparisons
- **GFLOP/s**: For matmul, indicates compute efficiency
- **Tokens/sec**: End-to-end throughput for inference

### Common Issues

1. **High Variance**: GPU thermal throttling, background processes
2. **First Run Slow**: Shader compilation, warmup needed
3. **Memory Pressure**: Large allocations may cause swapping

## Hardware Notes

### Apple Silicon (M1/M2/M3)
- Use Metal backend (default)
- Good FP32 performance, FP16 via shader conversion
- Limited subgroup operations

### Windows/Linux (Vulkan)
- Enable `--enable-features=Vulkan`
- NVIDIA has best performance
- AMD may need latest drivers

### SwiftShader (CI)
- Software rendering, ~100x slower
- Good for correctness testing
- Not representative of real performance

## Adding New Benchmarks

1. Add a benchmark helper in `tests/kernels/browser/test-page.js`
2. Use `KernelBenchmark` for warmup/iteration counts
3. Report median time and derived metrics in the console

Example:
```javascript
const stats = await benchmark.runBenchmark(
  async () => { /* kernel call */ },
  { warmupRuns: 5, timedRuns: 20, label: 'my_kernel' }
);
console.log(`Median: ${stats.medianMs}ms`);
```

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes and the OPFS purge helper.
