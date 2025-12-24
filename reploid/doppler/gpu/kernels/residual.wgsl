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

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.size) {
        return;
    }
    output[idx] = a[idx] + b[idx];
}

// In-place version: output = output + b
// Note: Caller should copy 'a' to 'output' first, then call this kernel
// This avoids requiring a different bind group layout with read_write on 'a'
@compute @workgroup_size(256, 1, 1)
fn add_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.size) {
        return;
    }
    output[idx] = output[idx] + b[idx];
}

// Vectorized version for better throughput
@compute @workgroup_size(64, 1, 1)
fn add_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x * 4u;
    let size = uniforms.size;

    if (idx >= size) {
        return;
    }

    // Handle up to 4 elements at a time
    let remaining = min(4u, size - idx);

    if (remaining >= 4u) {
        output[idx] = a[idx] + b[idx];
        output[idx + 1u] = a[idx + 1u] + b[idx + 1u];
        output[idx + 2u] = a[idx + 2u] + b[idx + 2u];
        output[idx + 3u] = a[idx + 3u] + b[idx + 3u];
    } else {
        for (var i = 0u; i < remaining; i = i + 1u) {
            output[idx + i] = a[idx + i] + b[idx + i];
        }
    }
}

// Fused residual + scale: output = a + scale * b
@compute @workgroup_size(256, 1, 1)
fn add_scaled(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.size) {
        return;
    }
    output[idx] = a[idx] + uniforms.scale * b[idx];
}
