// bias_add_backward.wgsl
//
// Computes gradient for bias: dB = sum_over_tokens(dY)
//
// For dY [num_tokens, dim], result dB is [dim].

struct Uniforms {
    num_tokens: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_bias: array<f32>;

override WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= u.dim) {
        return;
    }

    var sum: f32 = 0.0;
    for (var t: u32 = 0u; t < u.num_tokens; t = t + 1u) {
        sum = sum + grad_output[t * u.dim + d];
    }

    grad_bias[d] = sum;
}
