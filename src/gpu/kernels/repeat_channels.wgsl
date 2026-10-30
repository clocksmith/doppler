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
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let spatial = u.height * u.width;
    let out_channels = u.in_channels * u.repeats;
    let count = out_channels * spatial;
    let group = wid.x + wid.y * grid.x;
    if (group >= (count - 1u) / WORKGROUP_SIZE + 1u) {
        return;
    }
    let idx = group * WORKGROUP_SIZE + lid.x;
    if (idx >= count) {
        return;
    }

    let out_channel = idx / spatial;
    let spatial_idx = idx % spatial;
    let channel = out_channel / u.repeats;
    output[idx] = input[channel * spatial + spatial_idx];
}
