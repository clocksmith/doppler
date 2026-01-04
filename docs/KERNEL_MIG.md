# Kernel Migration Plan: 1 Entry Point Per File

## Goal

Refactor all 44 WGSL kernels to follow strict "1 entry point per file" pattern, using `override` constants and uniforms for all parameterization.

**Benefits:**
- Simpler mental model: 1 file = 1 kernel = 1 pipeline
- Easier testing/debugging (isolate issues to single file)
- Clearer naming convention
- Reduced code duplication via override-based specialization

---

## Complete Kernel Migration Table

### Attention Family (9 files -> 3 files)

| Current File | Entry Points | Migration |
|--------------|--------------|-----------|
| `attention.wgsl` | 2 (main, attention_decode) | -> `attention_prefill.wgsl` |
| `attention_f16kv.wgsl` | 2 (main, attention_decode) | Merge into `attention_prefill.wgsl` via `override F16_KV: bool` |
| `attention_small.wgsl` | 1 | Merge into `attention_prefill.wgsl` via `override SMALL_TILES: bool` |
| `attention_small_f16kv.wgsl` | 1 | Merge into `attention_prefill.wgsl` via override combo |
| `attention_streaming.wgsl` | 1 | -> `attention_streaming.wgsl` (keep separate, too different) |
| `attention_streaming_f16kv.wgsl` | 1 | Merge into `attention_streaming.wgsl` via `override F16_KV: bool` |
| `attention_decode_optimized.wgsl` | 3 (main, main_multihead, main_f16kv) | -> `attention_decode.wgsl` |
| `attention_decode_subgroup.wgsl` | 1 | Merge into `attention_decode.wgsl` via `override USE_SUBGROUPS: bool` |
| `attention_decode_chunked_f16kv.wgsl` | 1 | Merge into `attention_decode.wgsl` |

**Result:** 9 files -> 3 files (`attention_prefill.wgsl`, `attention_decode.wgsl`, `attention_streaming.wgsl`)

**New overrides for `attention_prefill.wgsl`:**
```wgsl
override F16_KV: bool = false;
override SMALL_TILES: bool = false;  // 32 vs 64 tile
override BLOCK_SIZE: u32 = 64u;
override HEAD_TILE: u32 = 64u;
```

**New overrides for `attention_decode.wgsl`:**
```wgsl
override F16_KV: bool = false;
override USE_SUBGROUPS: bool = true;
override MULTIHEAD: bool = false;  // 4 heads per WG
```

---

### Activation Kernels (1 file -> 3 files)

| Current File | Entry Points | Migration |
|--------------|--------------|-----------|
| `silu.wgsl` | 16 | Split by algorithm |

**silu.wgsl current entries:**
1. `main` - basic SiLU
2. `silu_bias` - SiLU + bias
3. `silu_gate` - SwiGLU: SiLU(gate) * up
4. `silu_gate_interleaved` - interleaved layout
5. `silu_gate_split` - split layout
6. `silu_vec4` - vectorized SiLU
7. `silu_gate_vec4` - vectorized SwiGLU
8. `silu_inplace` - in-place SiLU
9. `gelu` - GELU activation
10. `geglu` - GeGLU: GELU(gate) * up
11. `relu` - ReLU
12. `leaky_relu` - Leaky ReLU
13. `silu_mul` - SiLU(a) * b
14. `silu_gate_rowsplit` - row-split SwiGLU
15. `geglu_rowsplit` - row-split GeGLU
16. `silu_batched` - batched SiLU

**Split into:**

| New File | Entry Point | Parameterization |
|----------|-------------|------------------|
| `silu.wgsl` | `main` | uniform: `has_gate`, `has_bias`, `layout` (0=separate, 1=interleaved, 2=split, 3=rowsplit) |
| `gelu.wgsl` | `main` | uniform: `has_gate`, `layout` |
| `relu.wgsl` | `main` | uniform: `leaky_alpha` (0.0 = regular ReLU) |

**Override for all:**
```wgsl
override USE_VEC4: bool = false;
```

---

### RoPE (1 file -> 3 files)

| Current File | Entry Points | Migration |
|--------------|--------------|-----------|
| `rope.wgsl` | 6 | Split by algorithm |

**rope.wgsl current entries:**
1. `main` - precomputed freqs
2. `rope_compute_freqs` - compute on-the-fly
3. `rope_qk` - fused Q+K RoPE
4. `precompute_freqs` - init freq table
5. `rope_ntk_scaled` - NTK scaling
6. `rope_yarn` - YaRN scaling

**Split into:**

| New File | Entry Point | Parameterization |
|----------|-------------|------------------|
| `rope_precomputed.wgsl` | `main` | Uses precomputed sin/cos tables |
| `rope_compute.wgsl` | `main` | uniform: `scaling_mode` (0=linear, 1=ntk, 2=yarn), `fused_qk` |
| `rope_init.wgsl` | `main` | Precompute frequency tables |

---

### Sampling (1 file -> 2 files)

| Current File | Entry Points | Migration |
|--------------|--------------|-----------|
| `sample.wgsl` | 6 | Split by algorithm |

**sample.wgsl current entries:**
1. `argmax` - greedy decode phase 1
2. `argmax_reduce` - greedy decode phase 2
3. `find_topk_phase1` - top-k phase 1
4. `find_topk_phase2` - top-k phase 2
5. `softmax_and_sample` - sample from distribution
6. `sample_single_pass` - small vocab fast path

**Split into:**

| New File | Entry Point | Parameterization |
|----------|-------------|------------------|
| `sample_argmax.wgsl` | `main` | `override IS_REDUCE_PHASE: bool = false` |
| `sample_topk.wgsl` | `main` | uniform: `phase` (0=find_topk_phase1, 1=find_topk_phase2, 2=softmax_and_sample) |

---

### Softmax (1 file -> 2 files)

| Current File | Entry Points | Migration |
|--------------|--------------|-----------|
| `softmax.wgsl` | 5 | Consolidate + split |

**softmax.wgsl current entries:**
1. `main` - standard softmax
2. `softmax_small` - size <= 256 fast path
3. `softmax_online` - online algorithm
4. `softmax_inplace` - in-place
5. `log_softmax` - log softmax

**Split into:**

| New File | Entry Point | Parameterization |
|----------|-------------|------------------|
| `softmax.wgsl` | `main` | `override SMALL_SIZE: bool`, `override ONLINE: bool`; uniform: `inplace` |
| `softmax_log.wgsl` | `main` | Log softmax (different math, keep separate) |

---

### Normalization (1 file -> 1 file)

| Current File | Entry Points | Migration |
|--------------|--------------|-----------|
| `rmsnorm.wgsl` | 4 | Consolidate via overrides |

**rmsnorm.wgsl current entries:**
1. `main` - general RMSNorm
2. `rmsnorm_small` - size <= 256
3. `rmsnorm_with_prenorm` - outputs prenorm value
4. `rmsnorm_inplace_residual` - cached input variant

**Keep as 1 file with parameterization:**
```wgsl
override SMALL_SIZE: bool = false;
override CACHE_INPUT: bool = false;
// uniform: has_residual, output_prenorm
```

---

### Matmul Family (7 files -> 4 files)

| Current File | Entry Points | Migration |
|--------------|--------------|-----------|
| `matmul_f32.wgsl` | 1 | -> `matmul.wgsl` |
| `matmul_f16.wgsl` | 2 (main, main_vec4) | Merge into `matmul.wgsl` via `override F16_MODE: bool` |
| `matmul_f16w_f32a.wgsl` | 1 | Merge into `matmul.wgsl` via `override MIXED_PRECISION: bool` |
| `matmul_f16w_f32a_naive.wgsl` | 1 | Delete (superseded by above) |
| `matmul_gemv.wgsl` | 1 | -> `matmul_gemv.wgsl` (keep, M=1 specialization) |
| `matmul_gemv_subgroup.wgsl` | 3 (main, multicol, batched) | Merge into `matmul_gemv.wgsl` via overrides |
| `matmul_gemv_residual.wgsl` | 2 | -> `matmul_gemv_residual.wgsl` (fused, keep separate) |

**New overrides for `matmul.wgsl`:**
```wgsl
override F16_MODE: bool = false;
override MIXED_PRECISION: bool = false;
override USE_VEC4: bool = false;
```

**New overrides for `matmul_gemv.wgsl`:**
```wgsl
override USE_SUBGROUPS: bool = true;
override MULTICOL: bool = false;
override BATCHED: bool = false;
```

---

### Quantization/Dequant (6 files -> 4 files)

| Current File | Entry Points | Migration |
|--------------|--------------|-----------|
| `dequant_shared.wgsl` | 3 (main, main_vec4, main_f16_out) | -> `dequant_q4k.wgsl` |
| `dequant_subgroup.wgsl` | 3 | Merge into `dequant_q4k.wgsl` via `override USE_SUBGROUPS: bool` |
| `dequant_f16_out.wgsl` | 2 | Merge via `override F16_OUT: bool` |
| `dequant_q6k.wgsl` | 1 | Keep separate (different format) |
| `dequant_q8_0.wgsl` | 1 | Keep separate (different format) |
| `dequant_mxfp4.wgsl` | 3 | Keep separate (different format, 2 uniform structs) |

**New overrides for `dequant_q4k.wgsl`:**
```wgsl
override USE_SUBGROUPS: bool = false;
override USE_VEC4: bool = false;
override F16_OUT: bool = false;
```

---

### Fused Kernels (4 files -> 4 files, fewer entries)

| Current File | Entry Points | Migration |
|--------------|--------------|-----------|
| `fused_ffn.wgsl` | 4 (main, main_multi, main_f16, main_batched) | Consolidate via overrides |
| `fused_matmul_q4.wgsl` | 3 (main, main_multicol, main_batched) | Consolidate via overrides |
| `fused_matmul_rmsnorm.wgsl` | 4 (main, gemv_rmsnorm_small/medium/phase1) | Consolidate via overrides |
| `fused_swiglu.wgsl` | 1 | Keep as-is |

**New overrides for `fused_ffn.wgsl`:**
```wgsl
override F16_WEIGHTS: bool = false;
override BATCHED: bool = false;
override MULTI_OUTPUT: bool = false;
```

**New overrides for `fused_matmul_q4.wgsl`:**
```wgsl
override MULTICOL: bool = false;
override BATCHED: bool = false;
```

**New overrides for `fused_matmul_rmsnorm.wgsl`:**
```wgsl
override SMALL_N: bool = false;
override MEDIUM_N: bool = false;
override PHASE1_ONLY: bool = false;
```

---

### MoE Kernels (3 files -> 3 files)

| Current File | Entry Points | Migration |
|--------------|--------------|-----------|
| `topk.wgsl` | 3 (main, topk_2_small, softmax_topk) | Consolidate |
| `moe_gather.wgsl` | 4 (count_and_map, gather_tokens, gather_single_pass, gather_tokens_vec4) | Keep multi-entry (phases) |
| `scatter_add.wgsl` | 4 (main, scatter_add_vec4, scatter_add_dynamic, scatter_add_accumulate) | Consolidate |

**topk.wgsl consolidation:**
```wgsl
override FUSED_SOFTMAX: bool = false;
override SMALL_K: bool = false;  // k=2, n<=8 optimization
```

**scatter_add.wgsl consolidation:**
```wgsl
override USE_VEC4: bool = false;
override DYNAMIC_LAYOUT: bool = false;
// uniform: accumulate (add to existing vs overwrite)
```

**moe_gather.wgsl exception:** Keep multi-entry (phases must run separately, different bind groups)

---

### Utility Kernels (10 files -> 8 files)

| Current File | Entry Points | Migration |
|--------------|--------------|-----------|
| `gather.wgsl` | 2 (main, gather_vec4) | Consolidate via `override USE_VEC4` |
| `gather_f16.wgsl` | 2 | Merge into `gather.wgsl` via `override F16_EMBEDDINGS` |
| `residual.wgsl` | 4 (main, add_inplace, add_vec4, add_scaled) | Consolidate |
| `scale.wgsl` | 2 (main, main_inplace) | Consolidate via uniform `inplace` |
| `split_qkv.wgsl` | 2 | Keep (different algorithms) |
| `check_stop.wgsl` | 1 | Keep as-is (already 1 entry) |
| `bias_add.wgsl` | 1 | Keep as-is |
| `cast_f32_to_f16.wgsl` | 1 | Keep as-is |
| `bf16_to_f32.wgsl` | 2 | Consolidate via override |
| `bf16_to_f16.wgsl` | 1 | Keep as-is |

**gather.wgsl consolidation:**
```wgsl
override USE_VEC4: bool = false;
override F16_EMBEDDINGS: bool = false;
```

**residual.wgsl consolidation:**
```wgsl
override USE_VEC4: bool = false;
// uniform: inplace, scale (0.0 = no scaling)
```

---

## Summary

| Category | Current Files | New Files | Reduction |
|----------|---------------|-----------|-----------|
| Attention | 9 | 3 | -67% |
| Activation | 1 | 3 | +200% (but 16->3 entries) |
| RoPE | 1 | 3 | +200% (but 6->3 entries) |
| Sampling | 1 | 2 | +100% (but 6->2 entries) |
| Softmax | 1 | 2 | +100% (but 5->2 entries) |
| RMSNorm | 1 | 1 | 0% (4->1 entries) |
| Matmul | 7 | 4 | -43% |
| Dequant | 6 | 4 | -33% |
| Fused | 4 | 4 | 0% (fewer entries) |
| MoE | 3 | 3 | 0% (fewer entries) |
| Utility | 10 | 8 | -20% |
| **Total** | **44** | **~37** | **-16%** |

**Entry point reduction:** 100+ entries -> ~40 entries (~60% reduction)

---

## Implementation Order

1. **Low-risk consolidations first:** `scale.wgsl`, `residual.wgsl`, `gather.wgsl`
2. **Activation split:** `silu.wgsl` -> 3 files (high value, many entries)
3. **Attention consolidation:** 9 -> 3 files (high complexity, test carefully)
4. **Matmul consolidation:** 7 -> 4 files
5. **Sampling/softmax:** Lower priority, working fine

---

## TypeScript Wrapper Updates

For each kernel migration, update the corresponding `.ts` wrapper:

1. **Pipeline creation:** Pass override constants via `device.createComputePipeline()`:
   ```typescript
   const pipeline = device.createComputePipeline({
     compute: {
       module: shaderModule,
       entryPoint: 'main',
       constants: {
         F16_KV: useF16KV ? 1 : 0,
         SMALL_TILES: useSmallTiles ? 1 : 0,
       }
     }
   });
   ```

2. **Uniform buffer:** Add fields for runtime parameterization (layout mode, flags)

3. **Pipeline cache:** Key by override constant values, not entry point names

---

## Testing Strategy

1. **Per-kernel tests:** Validate each override combination produces correct output
2. **Regression tests:** Compare output before/after migration for existing models
3. **Performance tests:** Ensure no regression from override branching (compile-time, no runtime cost)
