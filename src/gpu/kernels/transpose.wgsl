// transpose.wgsl

/**
 * Matrix transpose kernel.
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
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dispatch_stride = max(u._pad0, 1u);
    let row = gid.y;
    let col = gid.x + row * dispatch_stride;
    if (row >= u.rows || col >= u.cols) {
        return;
    }
    let idx = row * u.cols + col;
    let out_idx = col * u.rows + row;
    output[out_idx] = input[idx];
}
