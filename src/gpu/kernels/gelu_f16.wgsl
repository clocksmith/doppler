// GeLU Activation Kernel with F16 Input/Output
//
// F16 variant for reduced memory bandwidth when using F16 activations.
// Intermediate computations use F32 for precision.

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    size: u32,          // Total output elements
    rowsplit_dim: u32,  // Dim for rowsplit variants (0 when unused)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;
@group(0) @binding(3) var<storage, read> gate: array<f16>;

fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi: f32 = 0.7978845608;
    let c: f32 = 0.044715;
    let inner = sqrt_2_over_pi * (x + c * x * x * x);
    let inner_clamped = clamp(inner, -15.0, 15.0);
    return 0.5 * x * (1.0 + tanh(inner_clamped));
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
    output[idx] = f16(gelu(x));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn geglu_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if (idx >= u.size) {
        return;
    }

    let up = f32(input[idx]);
    let g = f32(gate[idx]);
    output[idx] = f16(gelu(g) * up);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn geglu_rowsplit_f16(
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

    output[idx] = f16(gelu(g) * up);
}
