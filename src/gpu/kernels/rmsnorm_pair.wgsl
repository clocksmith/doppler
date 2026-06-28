// Sandwich RMSNorm pair kernel.
//
// Computes:
//   post_output = residual + rmsnorm(input, post_weight)
//   pre_output  = rmsnorm(post_output, pre_weight)
//
// This is intended for decode sandwich-norm paths where post_attn_norm is
// immediately followed by pre_ffn_norm and both tensors are needed downstream.

enable subgroups;

override WORKGROUP_SIZE: u32 = 256u;
override RMS_NORM_OFFSET: bool = false;
override POST_WEIGHT_IS_F16: bool = false;
override PRE_WEIGHT_IS_F16: bool = false;

const MAX_WORKGROUP_SIZE: u32 = 256u;
const MAX_SUBGROUPS: u32 = 32u;
const MAX_PAIR_CACHE_SIZE: u32 = 4608u;

struct Uniforms {
    size: u32,
    num_tokens: u32,
    eps: f32,
    has_residual: u32,
    token_stride: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read> post_weight: array<u32>;
@group(0) @binding(4) var<storage, read> pre_weight: array<u32>;
@group(0) @binding(5) var<storage, read_write> post_output: array<f32>;
@group(0) @binding(6) var<storage, read_write> pre_output: array<f32>;

var<workgroup> shared_sum: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_post: array<f32, MAX_PAIR_CACHE_SIZE>;
var<workgroup> sg_partial_sums: array<f32, MAX_SUBGROUPS>;

fn apply_weight(w: f32) -> f32 {
    if (RMS_NORM_OFFSET) {
        return 1.0 + w;
    }
    return w;
}

fn load_post_weight(idx: u32) -> f32 {
    if (POST_WEIGHT_IS_F16) {
        let packed = post_weight[idx >> 1u];
        let pair = unpack2x16float(packed);
        return select(pair.x, pair.y, (idx & 1u) == 1u);
    }
    return bitcast<f32>(post_weight[idx]);
}

fn load_pre_weight(idx: u32) -> f32 {
    if (PRE_WEIGHT_IS_F16) {
        let packed = pre_weight[idx >> 1u];
        let pair = unpack2x16float(packed);
        return select(pair.x, pair.y, (idx & 1u) == 1u);
    }
    return bitcast<f32>(pre_weight[idx]);
}

fn token_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.y * max(u.token_stride, 1u) + wg_id.x;
}

fn reduce_workgroup_sum(local_sum: f32, thread_idx: u32) -> f32 {
    shared_sum[thread_idx] = local_sum;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }

    return shared_sum[0];
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

fn compute_post(base_offset: u32, idx: u32, input_inv_rms: f32) -> f32 {
    let source = input[base_offset + idx];
    var post = source * input_inv_rms * apply_weight(load_post_weight(idx));
    if (u.has_residual != 0u) {
        post = post + residual[base_offset + idx];
    }
    return post;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let token_idx = token_index(wg_id);
    let thread_idx = local_id.x;
    let size = u.size;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * size;
    let elements_per_thread = (size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    var input_sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let source = input[base_offset + idx];
            input_sum_sq = input_sum_sq + source * source;
        }
    }

    let input_total = reduce_workgroup_sum(input_sum_sq, thread_idx);
    let input_inv_rms = 1.0 / sqrt(input_total / f32(size) + u.eps);
    workgroupBarrier();

    var post_sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let post = compute_post(base_offset, idx, input_inv_rms);
            shared_post[idx] = post;
            post_output[base_offset + idx] = post;
            post_sum_sq = post_sum_sq + post * post;
        }
    }

    let post_total = reduce_workgroup_sum(post_sum_sq, thread_idx);
    let post_inv_rms = 1.0 / sqrt(post_total / f32(size) + u.eps);
    workgroupBarrier();

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let post = shared_post[idx];
            pre_output[base_offset + idx] = post * post_inv_rms * apply_weight(load_pre_weight(idx));
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_subgroup(
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

    var input_sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let source = input[base_offset + idx];
            input_sum_sq = input_sum_sq + source * source;
        }
    }

    let input_total = reduce_subgroup_sum(input_sum_sq, thread_idx, sg_lane, sg_size);
    let input_inv_rms = 1.0 / sqrt(input_total / f32(size) + u.eps);
    workgroupBarrier();

    var post_sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let post = compute_post(base_offset, idx, input_inv_rms);
            shared_post[idx] = post;
            post_output[base_offset + idx] = post;
            post_sum_sq = post_sum_sq + post * post;
        }
    }

    let post_total = reduce_subgroup_sum(post_sum_sq, thread_idx, sg_lane, sg_size);
    let post_inv_rms = 1.0 / sqrt(post_total / f32(size) + u.eps);
    workgroupBarrier();

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let post = shared_post[idx];
            pre_output[base_offset + idx] = post * post_inv_rms * apply_weight(load_pre_weight(idx));
        }
    }
}
