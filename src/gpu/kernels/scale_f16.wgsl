enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    count: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.count) {
        return;
    }
    output[idx] = f16(f32(input[idx]) * u.scale);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.count) {
        return;
    }
    output[idx] = f16(f32(input[idx]) * u.scale);
}
