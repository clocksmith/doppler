// SiLU (Swish) Activation Kernel with F16 Input/Output
//
// F16 variant for reduced memory bandwidth when using F16 activations.
// Intermediate computations use F32 for precision.

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    size: u32,          // Total output elements
    rowsplit_dim: u32,  // Dim for rowsplit variants (0 when unused)
    clamp_max: f32,     // SwiGLU clamp (0 = disabled)
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;
@group(0) @binding(3) var<storage, read> gate: array<f16>;

fn sigmoid(x: f32) -> f32 {
    let clamped = clamp(x, -15.0, 15.0);
    return 1.0 / (1.0 + exp(-clamped));
}

fn silu(x: f32) -> f32 {
    return x * sigmoid(x);
}

fn clamp_swiglu(x: f32) -> f32 {
    if (u.clamp_max <= 0.0) {
        return x;
    }
    return clamp(x, -u.clamp_max, u.clamp_max);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if (idx >= u.size) {
        return;
    }

    let x = f32(input[idx]);
    output[idx] = f16(silu(x));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_gate_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if (idx >= u.size) {
        return;
    }

    let up = f32(input[idx]);
    let g = f32(gate[idx]);
    output[idx] = f16(clamp_swiglu(silu(g) * up));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_vec4_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let base_idx = global_id.x * 4u;
    if (base_idx >= u.size) {
        return;
    }

    let remaining = min(4u, u.size - base_idx);
    for (var i: u32 = 0u; i < remaining; i = i + 1u) {
        let x = f32(input[base_idx + i]);
        output[base_idx + i] = f16(silu(x));
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_gate_rowsplit_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if (idx >= u.size || u.rowsplit_dim == 0u) {
        return;
    }

    let dim = u.rowsplit_dim;
    let token_idx = idx / dim;
    let dim_idx = idx % dim;

    let row_base = token_idx * dim * 2u;
    let g = f32(input[row_base + dim_idx]);
    let up = f32(input[row_base + dim + dim_idx]);

    output[idx] = f16(clamp_swiglu(silu(g) * up));
}
