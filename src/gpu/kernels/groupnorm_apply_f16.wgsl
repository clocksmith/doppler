// GroupNorm Apply Kernel (NCHW, f16)

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    channels: u32,
    height: u32,
    width: u32,
    num_groups: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f16>;
@group(0) @binding(4) var<storage, read> bias: array<f16>;
@group(0) @binding(5) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let spatial = u.height * u.width;
    let total = u.channels * spatial;
    if (idx >= total) {
        return;
    }

    let channel = idx / spatial;
    let channels_per_group = u.channels / u.num_groups;
    let group = channel / channels_per_group;
    let stat_idx = group * 2u;
    let mean = stats[stat_idx];
    let inv_std = stats[stat_idx + 1u];

    let value = (f32(input[idx]) - mean) * inv_std;
    let scaled = value * f32(weight[channel]) + f32(bias[channel]);
    output[idx] = f16(scaled);
}
