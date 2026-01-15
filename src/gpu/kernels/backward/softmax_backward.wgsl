/**
 * Softmax backward kernel.
 */

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
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
    let cols = u.cols;
    let row = idx / cols;
    let col = idx % cols;
    if (row >= u.rows) {
        return;
    }

    let base = row * cols;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < cols; i = i + 1u) {
        let s = input[base + i];
        sum = sum + s * grad_output[base + i];
    }

    let s = input[base + col];
    output[idx] = s * (grad_output[idx] - sum);
}
