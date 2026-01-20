# DOPPLER WGSL Style Guide

WebGPU Shading Language conventions for DOPPLER kernels.

## File Structure

```wgsl
// kernel_name.wgsl

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS (override at pipeline creation)
// ═══════════════════════════════════════════════════════════════════

override HIDDEN_SIZE: u32 = 2048u;       // From manifest
override HEAD_DIM: u32 = 64u;            // From manifest
override NUM_HEADS: u32 = 32u;           // From manifest
override WORKGROUP_SIZE: u32 = 256u;     // Device-tuned

// ═══════════════════════════════════════════════════════════════════
// UNIFORMS (set per-dispatch)
// ═══════════════════════════════════════════════════════════════════

struct Uniforms {
    seq_len: u32,      // Changes every inference
    start_pos: u32,    // Changes every decode step
    kv_len: u32,       // Changes as KV cache grows
    _pad: u32,         // Alignment padding (16-byte aligned)
}

@group(0) @binding(0) var<uniform> u: Uniforms;

// ═══════════════════════════════════════════════════════════════════
// BINDINGS (buffers)
// ═══════════════════════════════════════════════════════════════════

@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// ═══════════════════════════════════════════════════════════════════
// ENTRY POINT
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Kernel implementation
}
```

---

## Constants vs Uniforms

| Type | When Set | Example | Use For |
|------|----------|---------|---------|
| **Constants** (`override`) | Pipeline creation | `HIDDEN_SIZE` | Model dimensions, feature flags |
| **Uniforms** | Per-dispatch | `seq_len` | Values that change per-inference |

### Constants (use `override`)

Values known at model load time:

```wgsl
// Model dimensions - from manifest.json
override HIDDEN_SIZE: u32 = 2048u;
override HEAD_DIM: u32 = 64u;
override NUM_HEADS: u32 = 32u;
override NUM_KV_HEADS: u32 = 8u;
override INTERMEDIATE_SIZE: u32 = 8192u;
override VOCAB_SIZE: u32 = 32000u;

// Quantization params - from format spec
override BLOCK_SIZE: u32 = 256u;        // Q4K block size

// Device tuning - from capability detection
override WORKGROUP_SIZE: u32 = 256u;
override TILE_SIZE: u32 = 16u;

// Feature flags - from manifest
override SCALE_EMBEDDINGS: bool = false;
override RMS_NORM_OFFSET: bool = true;   // Gemma (1+weight) formula
override USE_CAUSAL_MASK: bool = true;
```

### Uniforms (use struct)

Values that change per-inference:

```wgsl
struct Uniforms {
    // Position tracking
    seq_len: u32,           // Number of tokens this pass
    start_pos: u32,         // Position in sequence (for RoPE)
    kv_len: u32,            // Current KV cache length

    // Runtime flags
    is_prefill: u32,        // 1 = prefill, 0 = decode

    // Padding for 16-byte alignment
    _pad: vec2<u32>,
}
```

### Runtime Flags

Runtime-toggled behavior MUST be expressed as explicit uniform fields (not padding).
Keep TS and WGSL layouts in lockstep, and avoid "hidden" flags that rely on `_pad`.

```wgsl
struct Uniforms {
    has_residual: u32,  // 0 or 1
    _pad: vec3<u32>,
}
```

```ts
const UNIFORM_LAYOUT = {
  hasResidual: { offset: 0, size: 4 },
  _pad: { offset: 4, size: 12 },
} as const;
```

---

## Binding Conventions

### Standard Layout

```wgsl
// Binding order: uniforms, inputs (read), outputs (read_write)
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
```

### With Multiple Weight Buffers

```wgsl
// FFN example
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gate_weights: array<f32>;
@group(0) @binding(3) var<storage, read> up_weights: array<f32>;
@group(0) @binding(4) var<storage, read> down_weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
```

### Quantized Weights

```wgsl
// Q4K weights are packed differently
struct Q4KBlock {
    d: f16,                    // Scale
    dmin: f16,                 // Min scale
    scales: array<u8, 12>,     // Per-group scales
    qs: array<u8, 128>,        // Quantized values
}

@group(0) @binding(2) var<storage, read> weights_q4k: array<Q4KBlock>;
```

---

## Entry Points vs Override Constants vs Uniforms

This is the key design decision for kernel variants.

**Strict Rule (The "Topology Test"):**
Do not fork entry points for data values, dimensions, or optional math operations. Only fork entry points when the **loop nesting order** or **synchronization strategy** changes.

**Config Reminder:** Runtime tunables live in config only. Do not add WGSL-side flags that are toggled via CLI/URL overrides.

| Mechanism | When Set | Perf Cost | Use For |
|-----------|----------|-----------|---------|
| **Entry points** | Pipeline creation | None (separate code) | **Different Algorithms** (Control flow/Sync changes) |
| **Override constants** | Pipeline creation | None (compiler eliminates) | **Parametrization** (Dims, flags, loop bounds) |
| **Uniforms** | Per-dispatch | Branch divergence | **Dynamic State** (Per-call values only) |

### Decision Tree

```
Does the Loop Nesting or Sync Strategy change?
├─ YES → Separate entry point (or separate file)
│        Examples: GEMV (1D) vs GEMM (2D), Shared Mem vs Subgroups
│
└─ NO → Use Override Constants (Parametrization)
        ├─ Is it known at pipeline creation (Model Load)?
        │  ├─ YES → Override constant
        │  │        Examples: HIDDEN_SIZE, WORKGROUP_SIZE, HAS_BIAS
        │  │
        │  └─ NO → Uniform (Per-dispatch)
        │           Examples: seq_len, start_pos
```

### USE Entry Points For (Topology Changes)

Forking entry points is a last resort. Use only when the algorithm structure is incompatible.

*   **Algorithm Topology:** Matrix Multiplication (2D tiled loops) vs Matrix-Vector (1D reduction).
*   **Hardware Strategy:** `workgroupBarrier()` (Shared Memory) vs `subgroupAdd()` (Intrinsics).
*   **Multi-phase Ops:** `topk_phase1` (local reduction) vs `topk_phase2` (global merge).

### USE Override Constants For (Parametrization)

Even if 10+ constants change (e.g., `WORKGROUP_SIZE`, `TILE_SIZE`, `UNROLL_FACTOR`), if the topology is the same, use overrides.

*   **Dimensions:** `HIDDEN_SIZE`, `NUM_HEADS`.
*   **Tuning:** `WORKGROUP_SIZE`, `TILE_SIZE`.
*   **Feature Flags:** `HAS_BIAS`, `HAS_GATE`, `USE_VEC4`.
    *   *Why:* `if (HAS_BIAS) { ... }` is eliminated by the compiler (dead code elimination).
*   **Layouts:** `LAYOUT_ID` (0=contiguous, 1=strided).
*   **Math Variants:** `RMS_NORM_OFFSET` (Gemma 1+w formula).

### USE Uniforms For

*   **Dynamic State:** `seq_len`, `start_pos`, `kv_len`.
*   **Sampling Params:** `temperature` (if it changes per step).
*   **Runtime Switches:** Only if the value *must* change without reloading the pipeline (rare).
```

### DON'T: Explode Entry Points for Feature Combinations

```wgsl
// ☒ BAD - 16 entry points for feature combinations
fn silu()
fn silu_bias()
fn silu_gate()
fn silu_gate_bias()
fn silu_vec4()
fn silu_gate_vec4()
// ... explosion of combinations

// ✓ GOOD - override constants, compiler eliminates dead code
override HAS_BIAS: bool = false;
override HAS_GATE: bool = false;
override USE_VEC4: bool = false;

fn main() {
    var x = load_input();
    if (HAS_BIAS) { x += bias; }      // Eliminated if false
    x = silu(x);
    if (HAS_GATE) { x *= gate; }      // Eliminated if false
    store_output(x);
}
```

### DON'T: Mix Activation Functions in One File

```wgsl
// ☒ BAD - different functions in one file
fn silu() { ... }
fn gelu() { ... }
fn relu() { ... }

// ✓ GOOD - separate files
// silu.wgsl, gelu.wgsl, relu.wgsl
```

### File Organization

| Pattern | File Structure |
|---------|----------------|
| Different activations | `silu.wgsl`, `gelu.wgsl`, `relu.wgsl` |
| Same op, different quant | `matmul_f32.wgsl`, `matmul_f16.wgsl`, `matmul_q4k.wgsl` |
| Fused ops | `fused_ffn.wgsl`, `fused_ffn_q4k.wgsl` |
| Hardware variants | Same file with `main()` and `main_subgroup()` |

---

## Pipeline Layout Rules

**One pipeline layout per file.** Each file maps to a single combination of bindings + workgroup size + capability requirements.

### Hard Constraints

1. **Binding element types are fixed at compile time**
   - `array<f16>` vs `array<f32>` outputs cannot be toggled by override
   - If you need one code path, use packed `u32` with explicit unpacking
   - Otherwise keep separate files (`matmul_f16.wgsl`, `matmul_f32.wgsl`)

2. **Override constants cannot be used for array lengths**
   - Workgroup arrays must use fixed sizes or a MAX size
   - If you need multiple tile sizes, compile fixed-size variants and select at runtime
   ```wgsl
   // BAD - array size from override
   var<workgroup> tile: array<f32, TILE_SIZE * TILE_SIZE>;

   // GOOD - fixed MAX size
   const MAX_TILE: u32 = 1024u;  // 32x32
   var<workgroup> tile: array<f32, MAX_TILE>;
   ```

3. **Workgroup size is part of the pipeline**
   - Different `@workgroup_size` values are separate pipelines
   - Use separate files or separate entrypoints with clear variant IDs

4. **Capability requirements are compile-time**
   - If any code path uses `enable f16` or `enable subgroups`, the pipeline requires that capability
   - Keep subgroup/f16 fallbacks as separate files when needed

### Layout Boundaries

Group kernels by these criteria:

| Operation | Separate Files For |
|-----------|-------------------|
| **Attention** | KV dtype (f32 vs f16kv), subgroup vs non-subgroup |
| **Matmul** | Binding types (f32, f16, f16w_f32a), GEMV vs GEMM |
| **Dequant** | Output dtype (f32 vs f16), quant format (Q4K, Q6K, Q8_0) |
| **Activations** | dtype (f32 vs f16); use overrides for gate/bias/vec4 within dtype |
| **Sampling** | Multi-phase ops with different bind group layouts |

### Baseline Kernels

Keep non-optimized baseline variants available for debugging:
- f32 (no shader-f16 requirement)
- Non-subgroup (no subgroup requirement)
- Non-fused (individual operations)

### Enforcement

Validate kernel registry and check for policy violations:

```bash
npm run kernels:check   # Registry validation + override-array lint
```

---

## Entry Points (Legacy Guidance)

### Single Entry (Preferred)

```wgsl
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Use override constants for variants
}
```

### Multiple Entries (When Justified)

```wgsl
// For M=1 vs M>1 (different algorithms)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main() {
    // GEMV - one row per workgroup
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main_batched() {
    // Tiled GEMM - different workgroup layout
}
```

---

## Workgroup Sizes

### Use Override (Tunable)

```wgsl
// GOOD - can be tuned per-device
override WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(...) { }
```

### DON'T Hardcode

```wgsl
// BAD - can't tune for different devices
@compute @workgroup_size(256, 1, 1)
fn main(...) { }
```

### Common Sizes by Operation

| Operation | Typical Size | Notes |
|-----------|--------------|-------|
| Matmul | 16x16 (256) | 2D tiling |
| GEMV (M=1) | 256x1 | 1D reduction |
| Attention | 32-64 | Memory-bound |
| RMSNorm | 256 | Reduction |
| Element-wise | 256 | Max occupancy |

---

## Naming Conventions

### File Names

```
{operation}_{variant}.wgsl

matmul_f16.wgsl         # F16 matmul
matmul_gemv.wgsl        # M=1 optimized
attention_causal.wgsl   # With causal mask
fused_ffn_swiglu.wgsl   # Fused operation
rmsnorm.wgsl            # Standard (no variant)
```

### Constants

```wgsl
// UPPER_SNAKE_CASE
override HIDDEN_SIZE: u32 = 2048u;
override NUM_HEADS: u32 = 32u;
override WORKGROUP_SIZE: u32 = 256u;
```

### Struct Fields

```wgsl
// snake_case for WGSL
struct Uniforms {
    seq_len: u32,
    start_pos: u32,
    num_tokens: u32,
}
```

### Functions

```wgsl
// snake_case
fn compute_attention_score(q: vec4<f32>, k: vec4<f32>) -> f32 { }
fn apply_rope(x: vec2<f32>, cos: f32, sin: f32) -> vec2<f32> { }
```

---

## Common Patterns

### Index Calculation

```wgsl
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // Bounds check
    if (idx >= u.seq_len * HIDDEN_SIZE) {
        return;
    }

    // Decompose into token and dimension
    let token_idx = idx / HIDDEN_SIZE;
    let dim_idx = idx % HIDDEN_SIZE;
}
```

### Tiled Access (Matmul)

```wgsl
override TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<f32, 256>;  // TILE_SIZE * TILE_SIZE
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row = wid.y * TILE_SIZE + lid.y;
    let col = wid.x * TILE_SIZE + lid.x;

    var sum = 0.0f;

    for (var t = 0u; t < K / TILE_SIZE; t++) {
        // Load tile to shared memory
        tile_a[lid.y * TILE_SIZE + lid.x] = A[row * K + t * TILE_SIZE + lid.x];
        tile_b[lid.y * TILE_SIZE + lid.x] = B[(t * TILE_SIZE + lid.y) * N + col];

        workgroupBarrier();

        // Compute partial sum
        for (var k = 0u; k < TILE_SIZE; k++) {
            sum += tile_a[lid.y * TILE_SIZE + k] * tile_b[k * TILE_SIZE + lid.x];
        }

        workgroupBarrier();
    }

    C[row * N + col] = sum;
}
```

### Reduction (RMSNorm)

```wgsl
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn rmsnorm(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let token_idx = gid.x / WORKGROUP_SIZE;
    let local_idx = lid.x;

    // Each thread sums a portion
    var local_sum = 0.0f;
    for (var i = local_idx; i < HIDDEN_SIZE; i += WORKGROUP_SIZE) {
        let val = input[token_idx * HIDDEN_SIZE + i];
        local_sum += val * val;
    }

    shared_sum[local_idx] = local_sum;
    workgroupBarrier();

    // Tree reduction
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (local_idx < stride) {
            shared_sum[local_idx] += shared_sum[local_idx + stride];
        }
        workgroupBarrier();
    }

    // Normalize
    let rms = sqrt(shared_sum[0] / f32(HIDDEN_SIZE) + 1e-6f);

    for (var i = local_idx; i < HIDDEN_SIZE; i += WORKGROUP_SIZE) {
        let val = input[token_idx * HIDDEN_SIZE + i];
        let w = weights[i];
        // Gemma uses (1 + weight) formula when RMS_NORM_OFFSET is true
        let weight_val = select(w, 1.0f + w, RMS_NORM_OFFSET);
        output[token_idx * HIDDEN_SIZE + i] = (val / rms) * weight_val;
    }
}
```

---

## Anti-Patterns

### DON'T: Use Magic Numbers

```wgsl
// BAD
let idx = gid.x * 2048 + lid.x;  // What is 2048?

// GOOD
let idx = gid.x * HIDDEN_SIZE + lid.x;
```

### DON'T: Hardcode Model Dimensions

```wgsl
// BAD - breaks for different models
const HEAD_DIM: u32 = 64u;

// GOOD - override from JavaScript
override HEAD_DIM: u32 = 64u;
```

### DON'T: Skip Bounds Checks

```wgsl
// BAD - out of bounds read
let val = input[gid.x];

// GOOD
if (gid.x < u.seq_len * HIDDEN_SIZE) {
    let val = input[gid.x];
}
```

### DON'T: Assume Alignment

```wgsl
// BAD - may not be 16-byte aligned
struct Uniforms {
    a: u32,
    b: u32,
    c: u32,
}

// GOOD - explicit padding
struct Uniforms {
    a: u32,
    b: u32,
    c: u32,
    _pad: u32,  // Align to 16 bytes
}
```

---

## See Also

- [JavaScript Style Guide](./JAVASCRIPT_STYLE_GUIDE.md) - Kernel wrapper conventions
- [General Style Guide](./GENERAL_STYLE_GUIDE.md) - General patterns

---

## Kernel Compatibility and Overrides

This section documents kernel path compatibility rules and runtime overrides.

### Runtime Kernel Paths (Config-Only)

Use runtime config to force kernel path selection:

```json
{
  "runtime": {
    "inference": {
      "kernelPath": "gemma2-q4k-fused-f16a"
    }
  }
}
```

Priority (low to high):
1. manifest `optimizations.kernelPath`
2. manifest `inference.defaultKernelPath`
3. runtime config `runtime.inference.kernelPath`

Kernel paths are explicit dispatch sequences. See `../CONFIG.md`.

### RDRR Layout vs Runtime Kernels

| RDRR Quantization | Layout Metadata | Runtime Kernel Mode | Requirements | Notes |
|---|---|---|---|---|
| F16 / BF16 | `defaultWeightLayout=row` or `column` | `gemma2-f16-f16a` or `gemma2-f16-f32a` | `shader-f16` for F16 | Layout affects transpose; kernel path controls arithmetic. |
| Q4_K_M | `q4kLayout=row_wise` | `gemma2-q4k-fused-f16a`, `gemma2-q4k-fused-f32a`, `gemma2-q4k-dequant-f16a/f32a` | `subgroups` for fused; `shader-f16` for F16 | Row-wise layout required for fused Q4K. |
| Q4_K_M | `q4kLayout=column_wise` | `gemma2-q4k-dequant-f16a/f32a` | `shader-f16` for F16 | Column-wise packs are not fused-compatible. |
| Q4_K_M | `q4kLayout=flat` | `gemma2-q4k-dequant-f16a/f32a` | `shader-f16` for F16 | Flat packing has no fused kernel. |
| MXFP4 | N/A | dequant + matmul (no dedicated kernel path yet) | `shader-f16` for F16 | Used for MoE experts; no fused matmul yet. |
| Q8_0 / Q8_K | N/A | dequant + matmul (planned) | `shader-f16` for F16 | Loader runtime kernels are planned; treat as packing only today. |

### OPFS Purge Helper

Manifest updates in OPFS require a purge to take effect:

```bash
doppler --config ./tmp-opfs-purge.json
```
