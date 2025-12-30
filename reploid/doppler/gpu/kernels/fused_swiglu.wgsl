/**
 * SwiGLU Activation Kernel (Row-Split Fused Gate/Up)
 *
 * Computes: output = SiLU(gate + gate_bias) * (up + up_bias)
 *
 * input layout: [numTokens, 2 * dim] flattened, row-major
 *   row = [gate[0..dim), up[0..dim)]
 * bias layout: [2 * dim]
 *   bias = [gate_bias[0..dim), up_bias[0..dim)]
 * output layout: [numTokens, dim]
 */

struct Uniforms {
    num_tokens: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn silu(x: f32) -> f32 {
    return x * sigmoid(x);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = uniforms.num_tokens * uniforms.dim;
    if (idx >= total) {
        return;
    }

    let token_idx = idx / uniforms.dim;
    let dim_idx = idx % uniforms.dim;

    let row_base = token_idx * uniforms.dim * 2u;
    let gate = input[row_base + dim_idx] + bias[dim_idx];
    let up = input[row_base + uniforms.dim + dim_idx] + bias[uniforms.dim + dim_idx];

    output[idx] = silu(gate) * up;
}

