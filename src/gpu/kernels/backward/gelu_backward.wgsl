// gelu_backward.wgsl

/**
 * GELU backward kernel.
 */

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.size) {
        return;
    }
    let x = input[idx];
    let sqrt_2_over_pi: f32 = 0.7978845608;
    let c: f32 = 0.044715;
    let x3 = x * x * x;
    let inner = sqrt_2_over_pi * (x + c * x3);
    let inner_clamped = clamp(inner, -15.0, 15.0);
    let tanh_inner = tanh(inner_clamped);
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let inner_deriv = sqrt_2_over_pi * (1.0 + 3.0 * c * x * x);
    let deriv = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * inner_deriv;
    output[idx] = grad_output[idx] * deriv;
}
