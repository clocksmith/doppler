// GroupNorm Stats Kernel (NCHW)
// Computes mean and inv-std for each group.

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
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;

var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_sq: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let group = wid.x;
    if (group >= u.num_groups) {
        return;
    }

    let channels_per_group = u.channels / u.num_groups;
    let group_size = channels_per_group * u.height * u.width;
    let base = group * group_size;

    var sum: f32 = 0.0;
    var sumsq: f32 = 0.0;

    var idx = lid.x;
    while (idx < group_size) {
        let value = input[base + idx];
        sum = sum + value;
        sumsq = sumsq + value * value;
        idx = idx + WORKGROUP_SIZE;
    }

    shared_sum[lid.x] = sum;
    shared_sq[lid.x] = sumsq;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (lid.x < stride) {
            shared_sum[lid.x] = shared_sum[lid.x] + shared_sum[lid.x + stride];
            shared_sq[lid.x] = shared_sq[lid.x] + shared_sq[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lid.x == 0u) {
        let count = f32(group_size);
        let mean = shared_sum[0] / count;
        let variance = (shared_sq[0] / count) - (mean * mean);
        let inv_std = 1.0 / sqrt(variance + u.eps);
        let offset = group * 2u;
        stats[offset] = mean;
        stats[offset + 1u] = inv_std;
    }
}
