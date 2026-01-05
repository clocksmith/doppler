# Matmul Kernel Internals

Technical deep-dive on matrix multiplication kernels, thread utilization, and GEMV variants.

---

## Fused Q4K Matmul Analysis

### Why Fused Q4K is SLOWER Than Dequant Path

**Benchmark Results (Dec 2025):**

| Path | Layout | tok/s | Notes |
|------|--------|-------|-------|
| **Dequant → F16 GEMV** | column_wise | **8.0** | **DEFAULT** |
| Dequant → F16 GEMV | flat | 7.0 | Good fallback |
| Fused Q4K kernel | row_wise | 3.0 | **2.7x SLOWER** |

### Root Cause: Poor Thread Utilization

The fused Q4K kernel (`matmul_q4_fused.wgsl`) has a fundamental design issue:

```
For Gemma 3 1B with K=1152:
- Q4K block size: 256 weights
- Blocks per row: ceil(1152/256) = 5 blocks
- Threads per workgroup: 256

Problem: 5 blocks ÷ 256 threads = 5 active threads
         251 of 256 threads (98%) are IDLE per workgroup!
```

**Why this happens:**
```wgsl
// matmul_q4_fused.wgsl (simplified)
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let block_idx = lid.x;  // 0-255
  let num_blocks = (K + 255) / 256;  // = 5 for K=1152

  if (block_idx >= num_blocks) { return; }  // 251 threads exit immediately!

  // Only 5 threads do actual work...
}
```

### Current Mitigation

- Converter defaults to `--q4k-layout column_wise`
- Loader uses dequant path via `kernelPlan.q4kStrategy = 'dequant_f16'`
- Fused kernel still available for future optimization

### Future Fix Options

1. **Redesign kernel:** Multiple blocks per thread (loop over blocks)
2. **2D workgroup:** Use [32, 8, 1] instead of [256, 1, 1]
3. **Different kernel for small K:** Switch strategy based on K dimension

---

## Kernel Utilization Audit

| Kernel | Workgroup | Active Threads | Utilization | Status |
|--------|-----------|----------------|-------------|--------|
| `q4_fused` | 256 | ceil(K/256) | **2%** (K=1152) | Deprecated |
| `q4_fused_multicol` | 256 | 32×8 | **100%** | Fixed |
| `q4_fused_batched` | 64×4 | 64×M | Varies | Audit |
| `gemv_subgroup` | 256 | 256 | **100%** | OK |
| `gemv_subgroup_multicol` | 256 | 256 | **100%** | OK |
| `dequant_q4k` | 64 | N×K/256 | **100%** | OK |
| `rmsnorm` | 256 | hidden_size | **100%** | OK |
| `silu` | 256 | N | **100%** | OK |

### How to Audit a Kernel

```
1. Find workgroup size: @compute @workgroup_size(X, Y, Z)
2. Count threads that exit early: if (id >= limit) { return; }
3. Calculate: utilization = active_threads / (X × Y × Z)
4. Fix if utilization < 50%
```

---

## F16 GEMV Multi-Column Kernel (LM Head)

For Gemma 3's 262K vocab LM head with F16 tied embeddings:
- Original `gemv_subgroup`: 4 columns/workgroup → 65,536 workgroups
- New `gemv_subgroup_multicol`: 32 columns/workgroup → 8,192 workgroups (8x fewer)

```
LM head: M=1, N=262144, K=1152
Weight size: 262144 × 1152 × 2 bytes (F16) = 603MB per token read
```

### Implementation

```typescript
// matmul.ts selection logic
if (N > MULTICOL_THRESHOLD) {  // MULTICOL_THRESHOLD = 8192
  variant = 'gemv_subgroup_multicol';
} else {
  variant = 'gemv_subgroup';
}

// Workgroup dispatch
if (variant === 'gemv_subgroup_multicol') {
  gemvWorkgroupsX = Math.ceil(N / 32);  // 32 columns per workgroup
}
```

### Findings

| Metric | Before (4-col) | After (32-col) | Change |
|--------|----------------|----------------|--------|
| Workgroups | 65,536 | 8,192 | -87% |
| Decode tok/s | ~7 | ~7 | ~0% |
| Per-token latency | ~140ms | ~140ms | ~0% |

### Analysis

The 8x reduction in workgroups did NOT improve performance:

1. **LM head is not the dominant bottleneck** - The 26 transformer layers have 4 matmuls each (Q/K/V/O projections) plus 2 FFN matmuls = 156 matmul operations per forward pass. The single LM head matmul is <5% of total time.

2. **Memory bandwidth limited** - 603MB weight read per token at theoretical 200GB/s = 3ms minimum. Observed ~140ms suggests compute or other overheads dominate.

---

## FFN Gate+Up Fusion

```typescript
// Before: 3 passes
gate = matmul(input, gateWeight)   // Pass 1
up = matmul(input, upWeight)       // Pass 2
out = matmul(silu(gate)*up, down)  // Pass 3

// After: 2 passes
gateUp = matmul(input, gateUpWeight)  // Pass 1 (fused)
out = matmul(silu_split(gateUp), down) // Pass 2
```

**Impact:** 1.2-1.3x FFN speedup (gate+up fused, then down)

---

## Key Files

| File | Purpose |
|------|---------|
| `gpu/kernels/matmul_q4_fused.wgsl` | Fused Q4K dequant+matmul (GEMV + multicol + batched) |
| `gpu/kernels/matmul_gemv_subgroup.wgsl` | F16 GEMV (4-col + 32-col multicol variants) |
| `gpu/kernels/matmul.ts` | Kernel selection, layout handling, multicol dispatch |
| `gpu/kernels/utils.ts` | Kernel configs (incl. `gemv_subgroup_multicol`) |
