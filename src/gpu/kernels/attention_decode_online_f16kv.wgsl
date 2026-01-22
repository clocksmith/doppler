// Online Decode Attention Kernel (f32 Q + f16 KV + f32 output)
//
// Uses online softmax with subgroup reductions and chunked KV processing.

enable f16;
enable subgroups;

const MAX_WORKGROUP_SIZE: u32 = 256u;
const MAX_SUBGROUPS: u32 = 256u;

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

var<workgroup> shared_q: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_scores: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> sg_max: array<f32, MAX_SUBGROUPS>;
var<workgroup> sg_sum: array<f32, MAX_SUBGROUPS>;
var<workgroup> global_max: f32;
var<workgroup> global_sum: f32;

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

    if (WORKGROUP_SIZE > MAX_WORKGROUP_SIZE || head_dim > WORKGROUP_SIZE) {
        return;
    }

    if (kv_len == 0u) {
        if (tid < head_dim) {
            let out_offset = head_idx * head_dim + tid;
            output[out_offset] = 0.0;
        }
        return;
    }

    let subgroup_id = tid / subgroup_size;
    let num_subgroups = (WORKGROUP_SIZE + subgroup_size - 1u) / subgroup_size;
    if (num_subgroups > MAX_SUBGROUPS) {
        return;
    }

    let kv_head_idx = get_kv_head_idx(head_idx);

    if (tid < head_dim) {
        let q_offset = head_idx * head_dim + tid;
        shared_q[tid] = Q[q_offset];
    }
    workgroupBarrier();

    var running_max: f32 = -3.402823e+38;
    var running_sum: f32 = 0.0;
    var out_accum: f32 = 0.0;

    var start_k: u32 = 0u;
    if (u.sliding_window > 0u && kv_len > u.sliding_window) {
        start_k = kv_len - u.sliding_window;
    }

    for (var k_start: u32 = start_k; k_start < kv_len; k_start = k_start + WORKGROUP_SIZE) {
        let k_pos = k_start + tid;
        let valid_k = k_pos < kv_len;
        var masked = false;
        var score: f32 = -3.402823e+38;

        if (valid_k) {
            let abs_key = u.kv_start + k_pos;
            masked = is_masked(abs_key);
            if (!masked) {
                let k_idx = get_kv_pos(k_pos);
                let k_offset = k_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim;
                var dot: f32 = 0.0;
                for (var d: u32 = 0u; d < head_dim; d = d + 2u) {
                    let q0 = shared_q[d];
                    let k0 = f32(K[k_offset + d]);
                    dot = dot + q0 * k0;
                    if (d + 1u < head_dim) {
                        let q1 = shared_q[d + 1u];
                        let k1 = f32(K[k_offset + d + 1u]);
                        dot = dot + q1 * k1;
                    }
                }
                score = dot * u.scale;
                if (u.attn_softcap > 0.0) {
                    score = tanh(score / u.attn_softcap) * u.attn_softcap;
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
        let rescale = exp(running_max - new_max);

        var exp_score: f32 = 0.0;
        if (valid_k && !masked) {
            exp_score = exp(score - new_max);
        }
        shared_scores[tid] = exp_score;

        var chunk_sum = subgroupAdd(exp_score);
        if (sg_tid == 0u && subgroup_id < num_subgroups) {
            sg_sum[subgroup_id] = chunk_sum;
        }
        workgroupBarrier();

        if (tid == 0u) {
            var s: f32 = 0.0;
            for (var i: u32 = 0u; i < num_subgroups; i++) {
                s = s + sg_sum[i];
            }
            global_sum = s;
        }
        workgroupBarrier();

        running_sum = running_sum * rescale + global_sum;
        running_max = new_max;

        if (tid < head_dim) {
            out_accum = out_accum * rescale;
            let chunk_len = min(WORKGROUP_SIZE, kv_len - k_start);
            for (var k: u32 = 0u; k < chunk_len; k = k + 1u) {
                let score_idx = k;
                let k_pos_inner = k_start + k;
                let v_idx = get_kv_pos(k_pos_inner);
                let v_offset = v_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
                let v_val = f32(V[v_offset]);
                out_accum = out_accum + shared_scores[score_idx] * v_val;
            }
        }
        workgroupBarrier();
    }

    if (tid < head_dim) {
        let out_offset = head_idx * head_dim + tid;
        let inv_sum = select(0.0, 1.0 / running_sum, running_sum > 0.0);
        output[out_offset] = out_accum * inv_sum;
    }
}
