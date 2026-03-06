override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    in_channels: u32,
    out_channels: u32,
    height: u32,
    width: u32,
    groups: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let spatial = u.height * u.width;
    let spatial_idx = gid.x;
    let out_channel = gid.y;
    if (spatial_idx >= spatial || out_channel >= u.out_channels) {
        return;
    }
    let y = spatial_idx / u.width;
    let x = spatial_idx - y * u.width;

    let in_per_group = u.in_channels / u.groups;
    let out_per_group = u.out_channels / u.groups;
    let group_idx = out_channel / out_per_group;
    let in_offset = group_idx * in_per_group;

    var sum: f32 = bias[out_channel];
    for (var i: u32 = 0u; i < in_per_group; i = i + 1u) {
        let input_idx = ((in_offset + i) * u.height + y) * u.width + x;
        let weight_idx = out_channel * in_per_group + i;
        sum = sum + input[input_idx] * weight[weight_idx];
    }

    output[out_channel * spatial + spatial_idx] = sum;
}
