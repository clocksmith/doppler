/**
 * Bias Add Kernel
 *
 * Adds per-channel bias to a 2D tensor in-place.
 *
 * data layout: [numTokens, dim] flattened
 * bias layout: [dim]
 */

struct Uniforms {
    total_elements: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
}

override WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.total_elements) {
        return;
    }

    let d = idx % u.dim;
    data[idx] = data[idx] + bias[d];
}

