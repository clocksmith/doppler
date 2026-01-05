/**
 * Residual Add Kernel (F16)
 *
 * Performs element-wise addition for residual connections.
 * output = a + b
 */

enable f16;

struct Uniforms {
    size: u32,     // Total number of elements
    scale: f32,    // Scale factor for add_scaled
    _pad1: u32,
    _pad2: u32,
}

override WORKGROUP_SIZE: u32 = 256u;
override WORKGROUP_SIZE_VEC4: u32 = 64u;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f16>;
@group(0) @binding(2) var<storage, read> b: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.size) {
        return;
    }
    output[idx] = f16(f32(a[idx]) + f32(b[idx]));
}

// Vectorized version for better throughput
@compute @workgroup_size(WORKGROUP_SIZE_VEC4, 1, 1)
fn add_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x * 4u;
    let size = u.size;

    if (idx >= size) {
        return;
    }

    let remaining = min(4u, size - idx);

    if (remaining >= 4u) {
        output[idx] = f16(f32(a[idx]) + f32(b[idx]));
        output[idx + 1u] = f16(f32(a[idx + 1u]) + f32(b[idx + 1u]));
        output[idx + 2u] = f16(f32(a[idx + 2u]) + f32(b[idx + 2u]));
        output[idx + 3u] = f16(f32(a[idx + 3u]) + f32(b[idx + 3u]));
    } else {
        for (var i = 0u; i < remaining; i = i + 1u) {
            output[idx + i] = f16(f32(a[idx + i]) + f32(b[idx + i]));
        }
    }
}
