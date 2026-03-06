enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dispatch_stride = max(u._pad0, 1u);
    let idx = gid.y * dispatch_stride + gid.x;
    if (idx >= u.size) {
        return;
    }
    output[idx] = max(input[idx], 0.0h);
}
