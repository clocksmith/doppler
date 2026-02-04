// Online Softmax Kernel (Subgroup Variants)
//
// Subgroup variants use subgroupMax/subgroupAdd for faster reductions.

enable subgroups;

override WORKGROUP_SIZE: u32 = 256u;
const MAX_SUBGROUPS: u32 = 32u;

struct Uniforms {
    inner_size: u32,
    outer_size: u32,
    temperature: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> sg_partial_max: array<f32, MAX_SUBGROUPS>;
var<workgroup> sg_partial_sum: array<f32, MAX_SUBGROUPS>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_subgroup(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let row_idx = wg_id.x;
    let thread_idx = local_id.x;
    let inner_size = u.inner_size;
    let temperature = u.temperature;

    if (row_idx >= u.outer_size) {
        return;
    }

    let base_offset = row_idx * inner_size;
    let subgroup_id = thread_idx / sg_size;
    let num_subgroups = (WORKGROUP_SIZE + sg_size - 1u) / sg_size;
    let elements_per_thread = (inner_size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    var local_max: f32 = -3.402823e+38;

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < inner_size) {
            let val = input[base_offset + idx] / temperature;
            local_max = max(local_max, val);
        }
    }

    let sg_max = subgroupMax(local_max);
    if (sg_lane == 0u && subgroup_id < num_subgroups) {
        sg_partial_max[subgroup_id] = sg_max;
    }
    workgroupBarrier();

    if (thread_idx == 0u) {
        var gmax: f32 = -3.402823e+38;
        for (var s = 0u; s < num_subgroups; s++) {
            gmax = max(gmax, sg_partial_max[s]);
        }
        sg_partial_max[0] = gmax;
    }
    workgroupBarrier();
    let global_max = sg_partial_max[0];

    var local_sum: f32 = 0.0;

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < inner_size) {
            let val = input[base_offset + idx] / temperature;
            let exp_val = exp(val - global_max);
            local_sum = local_sum + exp_val;
        }
    }

    let sg_sum = subgroupAdd(local_sum);
    if (sg_lane == 0u && subgroup_id < num_subgroups) {
        sg_partial_sum[subgroup_id] = sg_sum;
    }
    workgroupBarrier();

    if (thread_idx == 0u) {
        var gsum: f32 = 0.0;
        for (var s = 0u; s < num_subgroups; s++) {
            gsum += sg_partial_sum[s];
        }
        sg_partial_sum[0] = gsum;
    }
    workgroupBarrier();
    let global_sum = sg_partial_sum[0];
    let inv_sum = select(0.0, 1.0 / global_sum, global_sum > 0.0);

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < inner_size) {
            let val = input[base_offset + idx] / temperature;
            let exp_val = exp(val - global_max);
            output[base_offset + idx] = exp_val * inv_sum;
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn softmax_small_subgroup(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let row_idx = wg_id.x;
    let thread_idx = local_id.x;
    let inner_size = u.inner_size;
    let temperature = u.temperature;

    if (row_idx >= u.outer_size) {
        return;
    }

    let base_offset = row_idx * inner_size;
    let subgroup_id = thread_idx / sg_size;
    let num_subgroups = (WORKGROUP_SIZE + sg_size - 1u) / sg_size;

    var val: f32 = -3.402823e+38;
    if (thread_idx < inner_size) {
        val = input[base_offset + thread_idx] / temperature;
    }

    let sg_max = subgroupMax(val);
    if (sg_lane == 0u && subgroup_id < num_subgroups) {
        sg_partial_max[subgroup_id] = sg_max;
    }
    workgroupBarrier();

    if (thread_idx == 0u) {
        var gmax: f32 = -3.402823e+38;
        for (var s = 0u; s < num_subgroups; s++) {
            gmax = max(gmax, sg_partial_max[s]);
        }
        sg_partial_max[0] = gmax;
    }
    workgroupBarrier();
    let global_max = sg_partial_max[0];

    var exp_val: f32 = 0.0;
    if (thread_idx < inner_size) {
        exp_val = exp(val - global_max);
    }

    let sg_sum = subgroupAdd(exp_val);
    if (sg_lane == 0u && subgroup_id < num_subgroups) {
        sg_partial_sum[subgroup_id] = sg_sum;
    }
    workgroupBarrier();

    if (thread_idx == 0u) {
        var gsum: f32 = 0.0;
        for (var s = 0u; s < num_subgroups; s++) {
            gsum += sg_partial_sum[s];
        }
        sg_partial_sum[0] = gsum;
    }
    workgroupBarrier();
    let global_sum = sg_partial_sum[0];
    let inv_sum = select(0.0, 1.0 / global_sum, global_sum > 0.0);

    if (thread_idx < inner_size) {
        output[base_offset + thread_idx] = exp_val * inv_sum;
    }
}
