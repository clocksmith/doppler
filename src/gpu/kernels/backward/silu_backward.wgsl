/**
 * SiLU backward kernel.
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
    let sigmoid = 1.0 / (1.0 + exp(-x));
    let deriv = sigmoid * (1.0 + x * (1.0 - sigmoid));
    output[idx] = grad_output[idx] * deriv;
}
