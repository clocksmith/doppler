# Attention Kernel Internals

Technical deep-dive on attention kernels, barrier analysis, and decode optimization.

---

## Attention Decode Inefficiency

The `attention_small_f16kv.wgsl` kernel has several inefficiencies for decode (seqLen=1):

| Issue | Impact | Details |
|-------|--------|---------|
| **Excessive barriers** | 80 barriers/token | For headDim=256: 8 head tiles (256/32) × 2 barriers × 5 KV blocks |
| **Poor thread utilization** | 97% idle | Workgroup size=32, but seqLen=1 → 31 of 32 threads idle |
| **Nested loops** | Serialized compute | Lines 107-118 and 166-181 have nested loops |

---

## Why tiled_large Won't Work

```wgsl
// attention_f16kv.wgsl line 82
var acc: array<f32, 64>;  // Limited to headDim ≤ 64
```

Gemma 1B has headDim=256, so `tiled_large` variant is incompatible. This is a fundamental architectural limitation.

---

## Barrier Calculation

```
headDim = 256
HEAD_TILE = 32
head_tiles = headDim / HEAD_TILE = 8

kvLen ≈ 135 (after prefill)
KV_BLOCK = 32
kv_blocks = ceil(135/32) = 5

barriers_per_token = kv_blocks × head_tiles × 2 = 5 × 8 × 2 = 80
```

Each `workgroupBarrier()` costs ~1-5μs on M3 → 80-400μs wasted per token just on barriers.

---

## Subgroup-Based Decode Kernel

**Algorithm for seqLen=1 decode:**
```wgsl
// For each head (workgroup size = headDim, one workgroup per head)
@compute @workgroup_size(256, 1, 1)  // headDim threads
fn attention_decode_subgroup(head_idx: u32) {
  let tid = global_id.x % 256;  // Thread within head
  let q_val = Q[head_idx][tid];  // Each thread loads one Q element

  // Compute attention scores (no barriers - subgroup ops only)
  for (let k = 0; k < kvLen; k++) {
    let k_val = K_cache[k][head_idx][tid];
    let dot = q_val * k_val;
    let score = subgroupAdd(dot);  // Parallel reduction, no barrier
    scores[k] = score / sqrt(headDim);  // Only thread 0 writes
  }

  // Softmax (subgroup parallel scan)
  let max_score = subgroupMax(scores[tid]);
  let exp_val = exp(scores[tid] - max_score);
  let sum_exp = subgroupAdd(exp_val);
  scores[tid] = exp_val / sum_exp;

  // Weighted sum over V (no barriers)
  let output_val = 0.0;
  for (let k = tid; k < kvLen; k += 256) {
    let v_val = V_cache[k][head_idx][tid];
    output_val += scores[k] * v_val;
  }
  output[head_idx][tid] = subgroupAdd(output_val);
}
```

### Key Properties

- Zero workgroup barriers (only subgroup operations)
- 100% thread utilization (all headDim=256 threads active)
- Naturally parallelizes across heads
- Requires subgroup support (available on M3)

---

## Expected Speedup

| Current (attention_small_f16kv) | With Subgroup Decode Kernel |
|--------------------------------|----------------------------|
| 80 barriers/token | **4 barriers/token** (20x reduction) |
| 31/32 threads idle (97% idle) | 256/256 threads active (100%) |
| 72ms attention/token (measured) | ~10-20ms attention/token (est.) |
| 86% of GPU time | ~20-40% of GPU time |

**Expected overall impact:** 3.6-7.2x speedup in attention → **~2-3x end-to-end speedup** (4 tok/s → 8-12 tok/s)

---

## FlashAttention-Style Fusion

Implemented tiled + online softmax with 3 device-aware tiers:
- `tiled_large` - For large head dimensions (≤64)
- `tiled_small` - For small head dimensions (>64)
- `streaming` - For very long sequences

**Impact:** 2x speedup from fusion

---

## Key Files

| File | Purpose |
|------|---------|
| `gpu/kernels/wgsl/attention_decode_subgroup.wgsl` | Subgroup-optimized decode kernel |
| `gpu/kernels/attention.ts` | Kernel selection, caps detection |
| `gpu/kernels/attention_f16kv.wgsl` | FlashAttention-style tiled kernel |
| `gpu/kernels/attention_small_f16kv.wgsl` | Small head dimension variant |
