// clamp.wgsl
// Clamp an f32 buffer in-place to [min, max].

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    size: u32,
    _pad0: u32,
    min: f32,
    max: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.size) {
        return;
    }
    data[idx] = clamp(data[idx], u.min, u.max);
}

