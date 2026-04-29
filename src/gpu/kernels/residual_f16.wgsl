// AUTO-GENERATED from src/gpu/kernels/residual.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// residual_f16.wgsl

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

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f16>;
@group(0) @binding(2) var<storage, read> b: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dispatch_stride = max(u._pad1, 1u);
    let idx = gid.y * dispatch_stride + gid.x;
    if (idx >= u.size) {
        return;
    }
    output[idx] = a[idx] + b[idx];
}
