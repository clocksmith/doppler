enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    in_channels: u32,
    height: u32,
    width: u32,
    repeats: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let spatial = u.height * u.width;
    let out_channels = u.in_channels * u.repeats;
    let total = out_channels * spatial;
    if (idx >= total) {
        return;
    }

    let out_channel = idx / spatial;
    let channel = out_channel / u.repeats;
    let spatial_idx = idx - out_channel * spatial;
    output[idx] = input[channel * spatial + spatial_idx];
}
