/**
 * Residual Add Kernel
 *
 * Performs element-wise addition for residual connections.
 * output = a + b
 */

struct Uniforms {
    size: u32,     // Total number of elements
    scale: f32,    // Scale factor for add_scaled
    _pad1: u32,
    _pad2: u32,
}

override WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.size) {
        return;
    }
    output[idx] = a[idx] + b[idx];
}

// In-place version: output = output + b
// Note: Caller should copy 'a' to 'output' first, then call this kernel
// This avoids requiring a different bind group layout with read_write on 'a'
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn add_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.size) {
        return;
    }
    output[idx] = output[idx] + b[idx];
}

// Fused residual + scale: output = a + scale * b
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn add_scaled(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.size) {
        return;
    }
    output[idx] = a[idx] + u.scale * b[idx];
}
