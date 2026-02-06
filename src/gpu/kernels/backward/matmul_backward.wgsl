// matmul_backward.wgsl

/**
 * Matmul backward kernel (placeholder).
 *
 * NOTE: The current implementation of matmul backward lives in
 * `matmul_backward.js`, which composes `transpose` + forward `matmul`
 * to compute:
 *   dX = dY @ W^T
 *   dW = X^T @ dY
 *
 * This WGSL entry point is kept as a stub for a future fused kernel.
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
    output[idx] = grad_output[idx];
}
