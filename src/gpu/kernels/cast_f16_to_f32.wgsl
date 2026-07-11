// cast_f16_to_f32.wgsl

/**
 * Cast F16 to F32 Kernel
 *
 * Converts a buffer of f16 values to f32.
 * Supports 2D dispatch for tensors larger than one workgroup dimension.
 */

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    let idx = gid.x + gid.y * num_wg.x * WORKGROUP_SIZE;
    if (idx >= u.count) {
        return;
    }
    output[idx] = f32(input[idx]);
}
