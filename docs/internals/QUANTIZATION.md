# Quantization Internals

Technical deep-dive on quantization formats, memory layouts, and performance implications.

---

## Q4K Block Layout

Q4K has 256-value super-blocks with embedded metadata:

```
Block (144 bytes): [d: f16, dmin: f16, scales: 12B, nibbles: 128B]
```

### Layout Options

**Flat packed (legacy):** Blocks cross row boundaries
```
Flat: [blk0][blk1][blk2][blk3][blk4][blk5]...
       ←─row 0──→←─row 0──→←row 1→←─row 1──→  ← WRONG!
```

**Column-wise Q4K:** Blocks organized by input column (FASTEST)
```
Col 0: [blk0][blk5][blk10]...  ← All K positions for output 0
Col 1: [blk1][blk6][blk11]...  ← All K positions for output 1
```

### Benchmark Results (Dec 2025)

| Layout | Decode tok/s | vs Baseline | Notes |
|--------|--------------|-------------|-------|
| **column_wise** | **8.0** | +14% | **DEFAULT - FASTEST** |
| flat | 7.0 | baseline | Simple packing |
| row_wise | 3.0 | -57% | Fused kernel has poor thread utilization |

---

## Why Column-Major is Faster (GPU Coalescing)

**The operation:** `output[1, out] = input[1, K] @ weight[out, K]^T`

When threads in a GPU warp access consecutive memory addresses, the hardware coalesces into a single transaction. Strided access splits into multiple transactions → high latency.

```
Row-major W[out, K]:
Thread 0 reads W[0, 0]    ← address 0
Thread 1 reads W[1, 0]    ← address K (strided - BAD)
Thread 2 reads W[2, 0]    ← address 2K

Column-major W^T[K, out]:
Thread 0 reads W^T[0, 0]  ← address 0
Thread 1 reads W^T[0, 1]  ← address 1 (contiguous - GOOD)
Thread 2 reads W^T[0, 2]  ← address 2
```

| Layout | Memory Pattern | GPU Coalescing | Performance |
|--------|----------------|----------------|-------------|
| Row-major W[out, K] | Row i contiguous | Threads read strided | Slower |
| Column-major W^T[K, out] | Column i contiguous | Threads read contiguous | **1.5-2x faster** |

### Why Column-Wise Q4K is Fastest for GEMV Decode

For GEMV (decode with M=1), computing `C[1, N] = A[1, K] × B[K, N]`:
- Each output column needs to read ALL K weights for that column
- Column-wise packing: column j's blocks are **contiguous in memory**
- Dequant kernel reads contiguous blocks → coalesced GPU access → high bandwidth

### Why Row-Wise is SLOWER

The fused Q4K kernel has 256 threads per workgroup. For K=1152:
- Only 5 Q4K blocks per row
- **251 of 256 threads are IDLE** → massive underutilization

---

## Precision Stack

```
Weights:     Q4_K_M (quantized, 4-bit) → keeps model size small
Matmul:      Fused Q4K kernel (dequant + matmul in one pass)
KV Cache:    F16 (not F32) → 2x memory savings
Activations: F16 where possible, F32 for numerically sensitive ops
Embeddings:  BF16 → F16 (converted at load time)
Norms:       BF16 → F32 (for numerical stability)
```

### WebGPU Precision Constraints

- WebGPU has **no native BF16 support** - must convert to F16 or F32
- F16 requires `shader-f16` feature (detected via `gpuCaps.hasF16`)
- Q4K fused kernel requires subgroup support (detected via `gpuCaps.hasSubgroups`)

### Expected Impact

| Change | Memory Savings | Speed Impact |
|--------|---------------|--------------|
| F16 KV cache | 2x KV memory | ~same |
| Fused Q4K | ~same | 1.3-1.5x faster |
| F16 activations | 2x activation memory | ~1.2x faster (bandwidth) |

---

## Available Quantization Functions

- `quantizeToQ4KMColumnWise(data, shape)` - Column-aligned Q4K blocks **(DEFAULT)**
- `quantizeToQ4KMRowWise(data, shape)` - Row-aligned Q4K blocks
- `quantizeToQ4KM(data, shape)` - Flat sequential packing
- `getQ4KSize(shape, layout)` - Calculate expected Q4K size

**Usage:**
```bash
# Convert with column-wise Q4K (default - fastest for GEMV decode)
npx tsx doppler/src/converter/node-converter.ts model/ output/ --quantize q4_k_m

# Explicitly specify layout
npx tsx doppler/src/converter/node-converter.ts model/ output/ --quantize q4_k_m --q4k-layout column_wise
```

---

## References

- [NVIDIA Efficient Matrix Transpose](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [WebGPU Matmul 1TFLOP Optimization](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel)
- [llama.cpp K-Quants Discussion](https://github.com/ggml-org/llama.cpp/discussions/5063)
