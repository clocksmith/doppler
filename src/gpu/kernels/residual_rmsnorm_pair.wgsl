// Residual + next-input RMSNorm pair kernel.
//
// Computes:
//   residual_output = input + residual
//   norm_output     = rmsnorm(residual_output, norm_weight)

enable subgroups;

override WORKGROUP_SIZE: u32 = 256u;
override RMS_NORM_OFFSET: bool = false;
override NORM_WEIGHT_IS_F16: bool = false;

const MAX_SUBGROUPS: u32 = 32u;
const MAX_PAIR_CACHE_SIZE: u32 = 4608u;

struct Uniforms {
    size: u32,
    num_tokens: u32,
    eps: f32,
    output_scale: f32,
    token_stride: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read> norm_weight: array<u32>;
@group(0) @binding(4) var<storage, read_write> residual_output: array<f32>;
@group(0) @binding(5) var<storage, read_write> norm_output: array<f32>;

var<workgroup> shared_residual: array<f32, MAX_PAIR_CACHE_SIZE>;
var<workgroup> sg_partial_sums: array<f32, MAX_SUBGROUPS>;

fn apply_weight(w: f32) -> f32 {
    if (RMS_NORM_OFFSET) {
        return 1.0 + w;
    }
    return w;
}

fn load_norm_weight(idx: u32) -> f32 {
    if (NORM_WEIGHT_IS_F16) {
        let packed = norm_weight[idx >> 1u];
        let pair = unpack2x16float(packed);
        return select(pair.x, pair.y, (idx & 1u) == 1u);
    }
    return bitcast<f32>(norm_weight[idx]);
}

fn token_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.y * max(u.token_stride, 1u) + wg_id.x;
}

fn reduce_subgroup_sum(local_sum: f32, thread_idx: u32, sg_lane: u32, sg_size: u32) -> f32 {
    let subgroup_id = thread_idx / sg_size;
    let num_subgroups = (WORKGROUP_SIZE + sg_size - 1u) / sg_size;
    let sg_sum = subgroupAdd(local_sum);

    if (sg_lane == 0u && subgroup_id < num_subgroups) {
        sg_partial_sums[subgroup_id] = sg_sum;
    }
    workgroupBarrier();

    if (thread_idx == 0u) {
        var sum: f32 = 0.0;
        for (var s = 0u; s < num_subgroups; s = s + 1u) {
            sum = sum + sg_partial_sums[s];
        }
        sg_partial_sums[0] = sum;
    }
    workgroupBarrier();

    return sg_partial_sums[0];
}

fn write_residual(base_offset: u32, idx: u32) -> f32 {
    let value = (input[base_offset + idx] + residual[base_offset + idx]) * u.output_scale;
    shared_residual[idx] = value;
    residual_output[base_offset + idx] = value;
    return value * value;
}

fn write_norm(base_offset: u32, idx: u32, inv_rms: f32) {
    let value = shared_residual[idx];
    norm_output[base_offset + idx] = value * inv_rms * apply_weight(load_norm_weight(idx));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let token_idx = token_index(wg_id);
    let thread_idx = local_id.x;
    let size = u.size;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * size;
    let elements_per_thread = (size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            sum_sq = sum_sq + write_residual(base_offset, idx);
        }
    }

    let total = reduce_subgroup_sum(sum_sq, thread_idx, sg_lane, sg_size);
    let inv_rms = 1.0 / sqrt(total / f32(size) + u.eps);
    workgroupBarrier();

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            write_norm(base_offset, idx, inv_rms);
        }
    }
}
