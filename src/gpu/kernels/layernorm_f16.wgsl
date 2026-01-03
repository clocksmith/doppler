// LayerNorm Kernel (F16 Input/Output)
//
// LayerNorm(x) = (x - mean) / sqrt(var + eps) * weight + bias

enable f16;

override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;
override PARAMS_IS_F16: bool = false; // Weight/bias packed as f16 pairs

struct Uniforms {
    size: u32,
    num_tokens: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> weight: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f16>;

var<workgroup> shared_sum: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_sum_sq: array<f32, MAX_WORKGROUP_SIZE>;

fn load_weight(idx: u32) -> f32 {
    if (PARAMS_IS_F16) {
        let packed = weight[idx >> 1u];
        let pair = unpack2x16float(packed);
        return select(pair.x, pair.y, (idx & 1u) == 1u);
    }
    return bitcast<f32>(weight[idx]);
}

fn load_bias(idx: u32) -> f32 {
    if (PARAMS_IS_F16) {
        let packed = bias[idx >> 1u];
        let pair = unpack2x16float(packed);
        return select(pair.x, pair.y, (idx & 1u) == 1u);
    }
    return bitcast<f32>(bias[idx]);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let token_idx = wg_id.x;
    let thread_idx = local_id.x;
    let size = u.size;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * size;

    var local_sum: f32 = 0.0;
    var local_sum_sq: f32 = 0.0;
    let elements_per_thread = (size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = f32(input[base_offset + idx]);
            local_sum = local_sum + x;
            local_sum_sq = local_sum_sq + x * x;
        }
    }

    shared_sum[thread_idx] = local_sum;
    shared_sum_sq[thread_idx] = local_sum_sq;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
            shared_sum_sq[thread_idx] = shared_sum_sq[thread_idx] + shared_sum_sq[thread_idx + stride];
        }
        workgroupBarrier();
    }

    let mean = shared_sum[0] / f32(size);
    let mean_sq = shared_sum_sq[0] / f32(size);
    let variance = max(0.0, mean_sq - mean * mean);
    let inv_std = 1.0 / sqrt(variance + u.eps);

    workgroupBarrier();

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = f32(input[base_offset + idx]);
            let norm = (x - mean) * inv_std;
            let y = norm * load_weight(idx) + load_bias(idx);
            output[base_offset + idx] = f16(y);
        }
    }
}
