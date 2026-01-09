# WGSL Kernels

41 WebGPU compute shaders for LLM inference.

## Categories

| Category | Count | Examples |
|----------|-------|----------|
| Attention | 9 | `attention.wgsl`, `attention_decode_*.wgsl` |
| Matmul | 8 | `matmul_f16.wgsl`, `fused_matmul_q4.wgsl`, `matmul_gemv*.wgsl` |
| Dequant | 5 | `dequant_q6k.wgsl`, `dequant_mxfp4.wgsl` |
| Fused | 4 | `fused_ffn.wgsl`, `fused_matmul_rmsnorm.wgsl`, `fused_matmul_q4.wgsl`, `fused_swiglu.wgsl` |
| Other | 16 | `rmsnorm.wgsl`, `rope.wgsl`, `softmax.wgsl`, `silu.wgsl` |

## Reusability Mechanisms

Three ways to make kernels flexible:

| Mechanism | When Set | Use For | Trade-off |
|-----------|----------|---------|-----------|
| **Entry points** | Pipeline creation | Different algorithms, workgroup sizes | Code duplication |
| **Override constants** | Pipeline creation | Parameterized array/workgroup sizes | Pipeline per config |
| **Uniforms** | Per dispatch | Dimensions, flags, runtime params | No compile-time optimization |

### Comparison

| Capability | Entry Points | Override Constants | Uniforms |
|------------|--------------|-------------------|----------|
| Array sizes | hardcoded | parameterized | no |
| Workgroup size | hardcoded | parameterized | no |
| Compiler optimization | full | full | branches only |
| Change per dispatch | select different | recompile | yes |
| Code duplication | high | minimal | none |

**DOPPLER uses entry points** over override constants - more code duplication but simpler pipeline management.

## Entry Points

One `.wgsl` file can have multiple `@compute` functions:

```wgsl
@compute @workgroup_size(256)
fn main() { ... }           // GEMV for small N

@compute @workgroup_size(256)
fn main_multicol() { ... }  // GEMV for large N (32 cols/workgroup)

@compute @workgroup_size(64, 4)
fn main_batched() { ... }   // Batched prefill (M > 1)
```

Selected at dispatch:
```javascript
pipeline = device.createComputePipeline({
  compute: { module, entryPoint: 'main_batched' }
});
```

## Uniforms

Runtime parameters passed per dispatch:

```wgsl
struct Uniforms {
    M: u32,              // Batch size
    N: u32,              // Output dimension
    K: u32,              // Inner dimension
    hasResidual: u32,    // Flag for conditional path
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

fn main() {
    // Use uniforms.M, uniforms.N for loop bounds
    if (uniforms.hasResidual == 1u) {
        // Conditional code path
    }
}
```

## When to Use What

| Scenario | Mechanism |
|----------|-----------|
| Different workgroup sizes for M=1 vs M>1 | Entry point |
| Different algorithms (GEMV vs GEMM) | Entry point |
| Variable dimensions (M, N, K) | Uniform |
| Optional feature (residual add, causal mask) | Uniform flag |
| Fixed tile size affecting shared memory | Entry point or override |

## Key Kernels

| Kernel | Entry Points | Purpose |
|--------|-------------|---------|
| `fused_matmul_q4.wgsl` | 3 | Q4_K quantized matmul (GEMV, multicol, batched) |
| `rmsnorm.wgsl` | 4 | RMSNorm with optional fused residual |
| `attention.wgsl` | 2 | Prefill attention (small/large) |
| `attention_decode_*.wgsl` | 1-3 | Decode attention variants |
| `silu.wgsl` | 5 | SiLU activation variants (gate, split, vec4, rowsplit) |
| `gelu.wgsl` | 3 | GeLU/GeGLU activation variants (gate, rowsplit) |

## Naming Conventions

- `fused_*.wgsl` - Multiple ops in one kernel (fused_ffn, fused_matmul_rmsnorm, etc.)
- `*_f16.wgsl` - F16 weights/activations
- `*_f32.wgsl` - F32 weights/activations
- `*_q4*.wgsl` - Q4_K quantized
- `*_subgroup.wgsl` - Uses subgroup operations
- `*_decode*.wgsl` - Optimized for M=1 decode
