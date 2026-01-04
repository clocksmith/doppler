// SiLU (Swish) Activation Kernel with F16 Input/Output
//
// F16 variant for reduced memory bandwidth when using F16 activations.
// Intermediate computations use F32 for precision.
//
// SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
//
// Includes fused variants:
// - SiLU(gate) * up (LLaMA SwiGLU FFN pattern)

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    size: u32,          // Total number of elements
    has_bias: u32,      // Repurposed: stores dim for rowsplit variants
    has_gate: u32,      // 1 if using gated variant
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;
@group(0) @binding(3) var<storage, read> gate: array<f16>;   // For gated variant

// Sigmoid helper with clamping (compute in F32)
fn sigmoid(x: f32) -> f32 {
    let clamped = clamp(x, -15.0, 15.0);
    return 1.0 / (1.0 + exp(-clamped));
}

// SiLU helper (compute in F32)
fn silu(x: f32) -> f32 {
    return x * sigmoid(x);
}

// Basic SiLU activation
// output = x * sigmoid(x)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    // Compute in F32 for precision
    let x = f32(input[idx]);
    output[idx] = f16(silu(x));
}

// Gated SiLU (SwiGLU): output = SiLU(gate) * up
// This is the pattern used in LLaMA FFN:
//   up = input @ W_up
//   gate = input @ W_gate
//   output = SiLU(gate) * up
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_gate_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    // Compute in F32 for precision
    let up = f32(input[idx]);
    let g = f32(gate[idx]);

    // SiLU(gate) * up
    output[idx] = f16(silu(g) * up);
}

// Row-split gated SiLU (for fused gate+up FFN)
// Input format: [numTokens, 2*dim] where each row is [gate[0..dim), up[0..dim)]
// Output format: [numTokens, dim]
// u.size = numTokens * dim (output size)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_gate_rowsplit_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    // size is the output size (numTokens * dim)
    if (idx >= u.size) {
        return;
    }

    // hasBias stores dim for rowsplit variant
    let dim = u.has_bias;
    let token_idx = idx / dim;
    let dim_idx = idx % dim;

    let row_base = token_idx * dim * 2u;
    let g = f32(input[row_base + dim_idx]);
    let up = f32(input[row_base + dim + dim_idx]);

    output[idx] = f16(silu(g) * up);
}

// Vectorized SiLU (process 4 elements per thread)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_vec4_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let base_idx = global_id.x * 4u;

    if (base_idx >= u.size) {
        return;
    }

    // Process up to 4 elements
    let remaining = min(4u, u.size - base_idx);

    for (var i: u32 = 0u; i < remaining; i = i + 1u) {
        let x = f32(input[base_idx + i]);
        output[base_idx + i] = f16(silu(x));
    }
}

// GELU activation for comparison (used in some models)
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi: f32 = 0.7978845608;
    let c: f32 = 0.044715;
    let inner = sqrt_2_over_pi * (x + c * x * x * x);
    let inner_clamped = clamp(inner, -15.0, 15.0);
    return 0.5 * x * (1.0 + tanh(inner_clamped));
}

// Row-split gated GELU (GeGLU) variant for models using GELU activation
// Input format: [numTokens, 2*dim] where each row is [gate[0..dim), up[0..dim)]
// Output format: [numTokens, dim]
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn geglu_rowsplit_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;

    if (idx >= u.size) {
        return;
    }

    let dim = u.has_bias;  // Repurposed: hasBias stores dim for rowsplit variant
    let token_idx = idx / dim;
    let dim_idx = idx % dim;

    let row_base = token_idx * dim * 2u;
    let g = f32(input[row_base + dim_idx]);
    let up = f32(input[row_base + dim + dim_idx]);

    output[idx] = f16(gelu(g) * up);
}
