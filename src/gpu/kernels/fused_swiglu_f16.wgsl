// fused_swiglu_f16.wgsl

/**
 * SwiGLU Activation Kernel (Row-Split Fused Gate/Up, f16 IO)
 */

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_tokens: u32,
    dim: u32,
    bias_offset: u32,
    clamp_max: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.num_tokens * u.dim;
    if (idx >= total) {
        return;
    }

    let token_idx = idx / u.dim;
    let dim_idx = idx % u.dim;
    let bias_base = u.bias_offset / 2u;

    let row_base = token_idx * u.dim * 2u;
    let gate = f32(input[row_base + dim_idx]) + f32(bias[bias_base + dim_idx]);
    let up = f32(input[row_base + u.dim + dim_idx]) + f32(bias[bias_base + u.dim + dim_idx]);

    output[idx] = f16(clamp_swiglu(silu(gate) * up));
}
