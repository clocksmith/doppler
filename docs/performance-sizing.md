# Performance and Sizing

This document is the canonical place for hardware sizing, memory planning, and performance expectations.

## Hardware tiers

### Tier 1: Unified memory

Examples:
- Apple Silicon (16GB+)
- Snapdragon X Elite
- future large-memory APUs

Typical characteristics:
- `memory64`
- `subgroups`
- `shader-f16`
- high effective memory bandwidth for large models

### Tier 2: Discrete memory64

Examples:
- RTX 3090/4090
- RTX 4080
- RX 7900 XTX

Typical characteristics:
- dedicated VRAM
- `memory64`
- good large-model support depending on quantization

### Tier 3: Basic WebGPU

Examples:
- entry-level integrated graphics
- older mid-range desktop GPUs

Typical characteristics:
- smaller buffer limits
- weaker throughput
- smaller viable model set

## Approximate model memory

| Model class | Typical quantization | Approx memory |
| --- | --- | --- |
| 1B | Q4_K | ~1.2 GB |
| 4B | Q4_K | ~2.8 GB |
| 12B | Q4_K | ~7.5 GB |
| 27B | Q4_K | ~16 GB |

These are planning numbers, not strict guarantees.

## Memory formula

```text
VRAM = (params_in_billions * bits_per_weight / 8) + kv_cache + activations
```

Quick approximations:

```text
Q4_K: VRAM_GB ~= params_B * 0.56 + 0.5
Q8:   VRAM_GB ~= params_B * 1.00 + 0.5
F16:  VRAM_GB ~= params_B * 2.00 + 0.5
```

## Performance expectations

Token speed and TTFT depend on:
- model size and quantization
- workload shape (prefill/decode)
- runtime config (`batchSize`, readback cadence)
- browser and driver stack

For claimable comparisons, use the benchmark flow in [benchmark-methodology.md](benchmark-methodology.md) and [../benchmarks/vendors/README.md](../benchmarks/vendors/README.md).

## Browser capability checks

```javascript
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
console.log('Max buffer size:', device.limits.maxBufferSize);
console.log('Max storage buffer:', device.limits.maxStorageBufferBindingSize);
```

## Related

- Setup workflow: [getting-started.md](getting-started.md)
- Operations troubleshooting: [operations.md](operations.md)
