override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    in_channels: u32,
    height: u32,
    width: u32,
    repeats: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let spatial = u.height * u.width;
    let out_channels = u.in_channels * u.repeats;
    let spatial_idx = gid.x;
    let out_channel = gid.y;
    if (out_channel >= out_channels || spatial_idx >= spatial) {
        return;
    }

    let channel = out_channel / u.repeats;
    let idx = out_channel * spatial + spatial_idx;
    output[idx] = input[channel * spatial + spatial_idx];
}
