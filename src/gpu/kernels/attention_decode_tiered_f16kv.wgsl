// Tiered Decode Attention Kernel (f32 Q + f16 KV)
//
// Hot ring buffer + cold paged cache with split-domain softmax.

enable f16;

const MAX_KV_LEN: u32 = 2048u;
const MAX_WORKGROUP_SIZE: u32 = 256u;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cold_len: u32,
    hot_len: u32,
    seq_len: u32,
    scale: f32,
    is_causal: u32,
    start_pos: u32,
    attn_softcap: f32,
    sliding_window: u32,
    hot_window: u32,
    hot_start: u32,
    cold_page_size: u32,
    cold_layout: u32,
    hot_layout: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K_hot: array<f16>;
@group(0) @binding(3) var<storage, read> V_hot: array<f16>;
@group(0) @binding(4) var<storage, read> K_cold: array<f16>;
@group(0) @binding(5) var<storage, read> V_cold: array<f16>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;
@group(0) @binding(7) var<storage, read> cold_page_table: array<u32>;

var<workgroup> shared_scores: array<f32, MAX_KV_LEN>;
var<workgroup> shared_partial: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;
var<workgroup> shared_acc: array<f32, MAX_WORKGROUP_SIZE>;

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

fn is_masked(abs_key: u32) -> bool {
    let abs_query = u.start_pos;
    if (u.is_causal != 0u && abs_key > abs_query) { return true; }
    if (u.sliding_window > 0u && abs_query >= u.sliding_window) {
        if (abs_key < abs_query - u.sliding_window + 1u) { return true; }
    }
    return false;
}

fn get_cold_pos(key_pos: u32) -> u32 {
    if (u.cold_layout == 2u) {
        let page_idx = key_pos / u.cold_page_size;
        let in_page = key_pos - (page_idx * u.cold_page_size);
        let phys_page = cold_page_table[page_idx];
        return phys_page * u.cold_page_size + in_page;
    }
    return key_pos;
}

fn get_hot_pos(hot_pos: u32) -> u32 {
    if (u.hot_layout == 1u && u.hot_window > 0u) {
        let abs_key = u.hot_start + hot_pos;
        return abs_key % u.hot_window;
    }
    return hot_pos;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let head_idx = workgroup_id.x;
    let tid = local_id.x;
    let head_dim = u.head_dim;
    let cold_len = u.cold_len;
    let hot_len = u.hot_len;
    let total_len = cold_len + hot_len;
    let scale = u.scale;
    let kv_head_idx = get_kv_head_idx(head_idx);

    if (WORKGROUP_SIZE > MAX_WORKGROUP_SIZE || head_dim > WORKGROUP_SIZE || total_len > MAX_KV_LEN) {
        return;
    }

    let valid = tid < head_dim;

    var q_val: f32 = 0.0;
    if (valid) {
        let q_offset = head_idx * head_dim + tid;
        q_val = Q[q_offset];
    }

    if (valid) {
        shared_acc[tid] = 0.0;
    }

    for (var k_pos: u32 = 0u; k_pos < cold_len; k_pos++) {
        var k_val: f32 = 0.0;
        if (valid) {
            let k_idx = get_cold_pos(k_pos);
            let k_offset = k_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
            k_val = f32(K_cold[k_offset]);
        }

        let partial = q_val * k_val;
        shared_partial[tid] = partial;
        workgroupBarrier();

        if (tid == 0u) {
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d++) {
                dot += shared_partial[d];
            }
            var s = dot * scale;
            if (u.attn_softcap > 0.0) {
                s = tanh(s / u.attn_softcap) * u.attn_softcap;
            }
            if (is_masked(k_pos)) {
                s = -3.402823e+38;
            }
            shared_scores[k_pos] = s;
        }
        workgroupBarrier();
    }

    for (var k_pos: u32 = 0u; k_pos < hot_len; k_pos++) {
        var k_val: f32 = 0.0;
        if (valid) {
            let k_idx = get_hot_pos(k_pos);
            let k_offset = k_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
            k_val = f32(K_hot[k_offset]);
        }

        let partial = q_val * k_val;
        shared_partial[tid] = partial;
        workgroupBarrier();

        if (tid == 0u) {
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d++) {
                dot += shared_partial[d];
            }
            var s = dot * scale;
            if (u.attn_softcap > 0.0) {
                s = tanh(s / u.attn_softcap) * u.attn_softcap;
            }
            let score_idx = cold_len + k_pos;
            let abs_key = u.hot_start + k_pos;
            if (is_masked(abs_key)) {
                s = -3.402823e+38;
            }
            shared_scores[score_idx] = s;
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        var max_score: f32 = -3.402823e+38;
        for (var k: u32 = 0u; k < total_len; k++) {
            max_score = max(max_score, shared_scores[k]);
        }
        shared_max = max_score;
    }
    workgroupBarrier();

    let max_score = shared_max;

    if (tid == 0u) {
        var sum_exp: f32 = 0.0;
        for (var k: u32 = 0u; k < total_len; k++) {
            let w = exp(shared_scores[k] - max_score);
            shared_scores[k] = w;
            sum_exp += w;
        }
        shared_sum = sum_exp;
    }
    workgroupBarrier();

    let sum_exp = shared_sum;
    let inv_sum = select(0.0, 1.0 / sum_exp, sum_exp > 0.0);

    if (valid) {
        var acc: f32 = 0.0;
        for (var score_idx: u32 = 0u; score_idx < total_len; score_idx++) {
            var v_val: f32 = 0.0;
            if (score_idx < cold_len) {
                let v_idx = get_cold_pos(score_idx);
                let v_offset = v_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
                v_val = f32(V_cold[v_offset]);
            } else {
                let hot_pos = score_idx - cold_len;
                let v_idx = get_hot_pos(hot_pos);
                let v_offset = v_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
                v_val = f32(V_hot[v_offset]);
            }
            acc += shared_scores[score_idx] * v_val;
        }
        shared_acc[tid] = acc * inv_sum;
    }
    workgroupBarrier();

    if (valid) {
        let out_offset = head_idx * head_dim + tid;
        output[out_offset] = shared_acc[tid];
    }
}
