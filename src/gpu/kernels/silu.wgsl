// SiLU (Swish) Activation Kernel
//
// SiLU(x) = x * sigmoid(x)
// Variants:
// - silu_gate: SiLU(gate) * up (separate gate buffer)
// - silu_gate_split: input = [gate..., up...]
// - silu_vec4: 4 elements per thread
// - silu_gate_rowsplit: input = [numTokens, 2*dim] row-split

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    size: u32,          // Total output elements
    rowsplit_dim: u32,  // Dim for rowsplit variants (0 when unused)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;

fn sigmoid(x: f32) -> f32 {
    let clamped = clamp(x, -15.0, 15.0);
    return 1.0 / (1.0 + exp(-clamped));
}

fn silu(x: f32) -> f32 {
    return x * sigmoid(x);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if (idx >= u.size) {
        return;
    }

    let x = input[idx];
    output[idx] = silu(x);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_gate(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if (idx >= u.size) {
        return;
    }

    let up = input[idx];
    let g = gate[idx];
    output[idx] = silu(g) * up;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_gate_split(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if (idx >= u.size) {
        return;
    }

    let g = input[idx];
    let up = input[idx + u.size];
    output[idx] = silu(g) * up;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_vec4(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let base_idx = global_id.x * 4u;
    if (base_idx >= u.size) {
        return;
    }

    let remaining = min(4u, u.size - base_idx);
    for (var i: u32 = 0u; i < remaining; i = i + 1u) {
        let x = input[base_idx + i];
        output[base_idx + i] = silu(x);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn silu_gate_rowsplit(
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
    let g = input[row_base + dim_idx];
    let up = input[row_base + dim + dim_idx];

    output[idx] = silu(g) * up;
}
