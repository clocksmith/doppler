// AUTO-GENERATED from src/gpu/kernels/attention_decode_online_f16.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// AUTO-GENERATED from src/gpu/kernels/attention_decode_online_f16.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// Online Decode Attention Kernel (f32 Q + f16 KV + f32 output)
//
// Uses online softmax with subgroup reductions and chunked KV processing.

enable f16;
enable subgroups;

const MAX_WORKGROUP_SIZE: u32 = 256u;
const MAX_SUBGROUPS: u32 = 256u;
const MAX_HEAD_DIM: u32 = 512u;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    seq_len: u32,
    scale: f32,
    is_causal: u32,
    start_pos: u32,
    attn_softcap: f32,
    sliding_window: u32,
    kv_len_source: u32,
    kv_start: u32,
    page_size: u32,
    kv_layout: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f16>;
@group(0) @binding(3) var<storage, read> V: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> kv_len_buffer: array<u32>;
@group(0) @binding(6) var<storage, read> page_table: array<u32>;

var<workgroup> shared_q: array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_scores: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> sg_max: array<f32, MAX_SUBGROUPS>;
var<workgroup> sg_sum: array<f32, MAX_SUBGROUPS>;
var<workgroup> global_max: f32;
var<workgroup> global_sum: f32;
var<workgroup> global_sum_correction: f32;

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

fn get_kv_len() -> u32 {
    if (u.kv_len_source == 0u) {
        return u.kv_len;
    }
    return kv_len_buffer[0];
}

fn is_masked(abs_key: u32) -> bool {
    let abs_query = u.start_pos;
    if (u.is_causal != 0u && abs_key > abs_query) { return true; }
    if (u.sliding_window > 0u && abs_query >= u.sliding_window) {
        if (abs_key < abs_query - u.sliding_window + 1u) { return true; }
    }
    return false;
}

fn get_kv_pos(key_pos: u32) -> u32 {
    let abs_key = u.kv_start + key_pos;
    if (u.kv_layout == 1u && u.sliding_window > 0u) {
        return abs_key % u.sliding_window;
    }
    if (u.kv_layout == 2u) {
        let page_idx = abs_key / u.page_size;
        let in_page = abs_key - (page_idx * u.page_size);
        let phys_page = page_table[page_idx];
        return phys_page * u.page_size + in_page;
    }
    return abs_key;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) sg_tid: u32,
) {
    let head_idx = workgroup_id.x;
    let tid = local_id.x;
    let head_dim = u.head_dim;
    let kv_len = get_kv_len();
    let q_offset = head_idx * head_dim;
    let out_dim0 = tid;
    let out_dim1 = tid + WORKGROUP_SIZE;
    let has_out_dim0 = out_dim0 < head_dim;
    let has_out_dim1 = out_dim1 < head_dim;

    if (WORKGROUP_SIZE > MAX_WORKGROUP_SIZE || head_dim > MAX_HEAD_DIM) {
        return;
    }

    if (kv_len == 0u) {
        if (has_out_dim0) {
            output[q_offset + out_dim0] = 0.0;
        }
        if (has_out_dim1) {
            output[q_offset + out_dim1] = 0.0;
        }
        return;
    }

    let subgroup_id = tid / subgroup_size;
    let num_subgroups = (WORKGROUP_SIZE + subgroup_size - 1u) / subgroup_size;
    if (num_subgroups > MAX_SUBGROUPS) {
        return;
    }

    let kv_head_idx = get_kv_head_idx(head_idx);

    if (has_out_dim0) {
        shared_q[out_dim0] = Q[q_offset + out_dim0];
    }
    if (has_out_dim1) {
        shared_q[out_dim1] = Q[q_offset + out_dim1];
    }
    workgroupBarrier();

    var running_max: f32 = -3.402823e+38;
    var running_sum: f32 = 0.0;
    var running_correction: f32 = 0.0;
    var out_accum0: f32 = 0.0;
    var out_correction0: f32 = 0.0;
    var out_accum1: f32 = 0.0;
    var out_correction1: f32 = 0.0;

    var start_k: u32 = 0u;
    if (u.sliding_window > 0u && kv_len > u.sliding_window) {
        start_k = kv_len - u.sliding_window;
        // Align to WORKGROUP_SIZE
        start_k = (start_k / WORKGROUP_SIZE) * WORKGROUP_SIZE;
    }

    for (var k_start: u32 = start_k; k_start < kv_len; k_start = k_start + WORKGROUP_SIZE) {
        let k_pos = k_start + tid;
        let valid_k = k_pos < kv_len;
        var masked = false;
        var score: f32 = -3.402823e+38;
        var score_correction: f32 = 0.0;

        if (valid_k) {
            let abs_key = u.kv_start + k_pos;
            masked = is_masked(abs_key);
            if (!masked) {
                let k_idx = get_kv_pos(k_pos);
                let k_offset = k_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim;
                var dot: f32 = 0.0;
                var correction: f32 = 0.0;
                for (var d: u32 = 0u; d < head_dim; d = d + 2u) {
                    let q0 = shared_q[d];
                    let k0 = f32(K[k_offset + d]);
                    let product0 = fma(q0, k0, 0.0);
                    let total0 = fma(1.0, dot, product0);
                    let error0 = select(fma(1.0, dot, fma(-1.0, total0, product0)),
                        fma(1.0, product0, fma(-1.0, total0, dot)), abs(dot) >= abs(product0));
                    correction = fma(1.0, correction, fma(1.0, fma(q0, k0, -product0), error0));
                    dot = total0;
                    if (d + 1u < head_dim) {
                        let q1 = shared_q[d + 1u];
                        let k1 = f32(K[k_offset + d + 1u]);
                        let product1 = fma(q1, k1, 0.0);
                    let total1 = fma(1.0, dot, product1);
                    let error1 = select(fma(1.0, dot, fma(-1.0, total1, product1)),
                        fma(1.0, product1, fma(-1.0, total1, dot)), abs(dot) >= abs(product1));
                    correction = fma(1.0, correction, fma(1.0, fma(q1, k1, -product1), error1));
                    dot = total1;
                    }
                }
                let dot_sum = fma(1.0, dot, correction);
                let dot_error = sum_error(dot, correction, dot_sum);
                score = dot_sum * u.scale;
                score_correction = dot_error * u.scale;
                let scale_error = fma(dot_sum, u.scale, -score);
                // Preserve the low term when the rounded high-product residual is zero.
                if (scale_error != 0.0) { score_correction += scale_error; }
                if (u.attn_softcap > 0.0) {
                    score = tanh(score / u.attn_softcap) * u.attn_softcap;
                    score_correction = 0.0;
                }
            }
        }

        var chunk_max = subgroupMax(score);
        if (sg_tid == 0u && subgroup_id < num_subgroups) {
            sg_max[subgroup_id] = chunk_max;
        }
        workgroupBarrier();

        if (tid == 0u) {
            var m: f32 = -3.402823e+38;
            for (var s: u32 = 0u; s < num_subgroups; s++) {
                m = max(m, sg_max[s]);
            }
            global_max = m;
        }
        workgroupBarrier();

        let chunk_max_val = global_max;
        let new_max = max(running_max, chunk_max_val);
        let rescale = exp_refined(running_max - new_max);

        var exp_score: f32 = 0.0;
        if (valid_k && !masked) {
            let difference = fma(-1.0, new_max, score);
            let error = sum_error(score, -new_max, difference);
            let low = fma(1.0, score_correction, error);
            let exponential = exp_refined(difference);
            exp_score = fma(exponential, low, exponential);
        }
        shared_scores[tid] = exp_score;

        workgroupBarrier();
        if (tid == 0u) {
            var sum = 0.0;
            var correction = 0.0;
            let count = min(WORKGROUP_SIZE, kv_len - k_start);
            for (var k = 0u; k < count; k++) {
                let value = shared_scores[k];
                let total = fma(1.0, sum, value);
                let error = select(fma(1.0, sum, fma(-1.0, total, value)),
                    fma(1.0, value, fma(-1.0, total, sum)), abs(sum) >= abs(value));
                correction = fma(1.0, correction, error);
                sum = total;
            }
            global_sum = sum;
            global_sum_correction = correction;
        }
        workgroupBarrier();

        let rescaled = fma(running_sum, rescale, 0.0);
        let rescale_error = fma(running_sum, rescale, -rescaled) + running_correction * rescale;
        let total_sum = fma(1.0, rescaled, global_sum);
        let sum_error = select(fma(1.0, rescaled, fma(-1.0, total_sum, global_sum)),
            fma(1.0, global_sum, fma(-1.0, total_sum, rescaled)), abs(rescaled) >= abs(global_sum));
        running_correction = fma(1.0, rescale_error, fma(1.0, global_sum_correction, sum_error));
        running_sum = total_sum;
        running_max = new_max;

        if (has_out_dim0 || has_out_dim1) {
            out_accum0 = fma(out_accum0, rescale, 0.0);
            out_correction0 = fma(out_correction0, rescale, 0.0);
            out_accum1 = fma(out_accum1, rescale, 0.0);
            out_correction1 = fma(out_correction1, rescale, 0.0);
            let chunk_len = min(WORKGROUP_SIZE, kv_len - k_start);
            for (var k: u32 = 0u; k < chunk_len; k = k + 1u) {
                let score_idx = k;
                let k_pos_inner = k_start + k;
                let v_idx = get_kv_pos(k_pos_inner);
                let v_base = v_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim;
                if (has_out_dim0) {
                    let value = f32(V[v_base + out_dim0]);
                    let probability = shared_scores[score_idx];
                    let product = fma(probability, value, 0.0);
                    let total = fma(1.0, out_accum0, product);
                    let error = select(fma(1.0, out_accum0, fma(-1.0, total, product)),
                        fma(1.0, product, fma(-1.0, total, out_accum0)), abs(out_accum0) >= abs(product));
                    out_correction0 = fma(1.0, out_correction0, fma(probability, value, -product) + error);
                    out_accum0 = total;
                }
                if (has_out_dim1) {
                    let value = f32(V[v_base + out_dim1]);
                    let probability = shared_scores[score_idx];
                    let product = fma(probability, value, 0.0);
                    let total = fma(1.0, out_accum1, product);
                    let error = select(fma(1.0, out_accum1, fma(-1.0, total, product)),
                        fma(1.0, product, fma(-1.0, total, out_accum1)), abs(out_accum1) >= abs(product));
                    out_correction1 = fma(1.0, out_correction1, fma(probability, value, -product) + error);
                    out_accum1 = total;
                }
            }
        }
        workgroupBarrier();
    }

    let inv_sum = select(0.0, reciprocal_refined(running_sum), running_sum > 0.0);
    if (has_out_dim0) {
        let quotient = out_accum0 * inv_sum;
        let residual = fma(-quotient, running_sum, out_accum0)
            + fma(-quotient, running_correction, out_correction0);
        output[q_offset + out_dim0] = fma(residual, inv_sum, quotient);
    }
    if (has_out_dim1) {
        let quotient = out_accum1 * inv_sum;
        let residual = fma(-quotient, running_sum, out_accum1)
            + fma(-quotient, running_correction, out_correction1);
        output[q_offset + out_dim1] = fma(residual, inv_sum, quotient);
    }
}

fn exp_refined(x: f32) -> f32 {
  if (abs(x) > 80.0) {
    return exp(x);
  }
  let n = round(x * 1.4426950408889634);
  var r = x;
  if (abs(x) >= 0.000000059604644775390625) {
    let bits = bitcast<u32>(abs(x));
    let mantissa = (bits & 0x007fffffu) | 0x00800000u;
    let shift = i32((bits >> 23u) & 255u) - 118;
    var input_fixed = 0u;
    if (shift >= 0) {
      input_fixed = mantissa << u32(shift);
    } else {
      let right = u32(-shift);
      input_fixed = (mantissa + (1u << (right - 1u))) >> right;
    }
    let power_fixed = u32(abs(n)) * 2977044472u;
    let residual = select(power_fixed - input_fixed, input_fixed - power_fixed, x >= 0.0);
    r = f32(bitcast<i32>(residual)) * 0.00000000023283064365386962890625;
  }
  var p = 0.0001984126984126984;
  p = fma(p, r, 0.001388888888888889);
  p = fma(p, r, 0.008333333333333333);
  p = fma(p, r, 0.041666666666666664);
  p = fma(p, r, 0.16666666666666666);
  p = fma(p, r, 0.5);
  p = fma(p, r, 1.0);
  p = fma(p, r, 1.0);
  return p * bitcast<f32>(u32(i32(n) + 127) << 23u);
}

fn reciprocal_refined(x: f32) -> f32 {
  let estimate = 1.0 / x;
  return fma(estimate, fma(-x, estimate, 1.0), estimate);
}

fn sum_error(a: f32, b: f32, rounded: f32) -> f32 {
    let a_is_high = abs(a) >= abs(b);
    let high = select(b, a, a_is_high);
    let low = select(a, b, a_is_high);
    return fma(-1.0, rounded, high) + low;
}
