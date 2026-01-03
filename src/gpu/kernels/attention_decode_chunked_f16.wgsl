// Chunked Decode Attention Kernel (f16 QKV + f16 output)
//
// Same as attention_decode_chunked_f16kv.wgsl but Q/K/V are f16 and output is f16.

enable f16;

const MAX_KV_LEN: u32 = 2048u;
const MAX_WORKGROUP_SIZE: u32 = 256u;

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
@group(0) @binding(1) var<storage, read> Q: array<f16>;
@group(0) @binding(2) var<storage, read> K: array<f16>;
@group(0) @binding(3) var<storage, read> V: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f16>;
@group(0) @binding(5) var<storage, read> kv_len_buffer: array<u32>;
@group(0) @binding(6) var<storage, read> page_table: array<u32>;

var<workgroup> shared_scores: array<f32, MAX_KV_LEN>;
var<workgroup> shared_partial: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;
var<workgroup> shared_acc: array<f32, MAX_WORKGROUP_SIZE>;

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

fn is_masked(key_pos: u32) -> bool {
    let abs_query = u.start_pos;
    let abs_key = u.kv_start + key_pos;
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

fn get_kv_len() -> u32 {
    if (u.kv_len_source == 0u) {
        return u.kv_len;
    }
    return kv_len_buffer[0];
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let head_idx = workgroup_id.x;
    let tid = local_id.x;
    let head_dim = u.head_dim;
    let kv_len = get_kv_len();
    let scale = u.scale;
    let kv_head_idx = get_kv_head_idx(head_idx);

    if (WORKGROUP_SIZE > MAX_WORKGROUP_SIZE || head_dim > WORKGROUP_SIZE || kv_len > MAX_KV_LEN) {
        return;
    }

    let valid = tid < head_dim;

    var q_val: f32 = 0.0;
    if (valid) {
        let q_offset = head_idx * head_dim + tid;
        q_val = f32(Q[q_offset]);
    }

    if (valid) {
        shared_acc[tid] = 0.0;
    }

    var start_k: u32 = 0u;
    if (u.sliding_window > 0u && kv_len > u.sliding_window) {
        start_k = kv_len - u.sliding_window;
    }

    for (var k_pos: u32 = start_k; k_pos < kv_len; k_pos++) {
        var k_val: f32 = 0.0;
        if (valid) {
            let k_idx = get_kv_pos(k_pos);
            let k_offset = k_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
            k_val = f32(K[k_offset]);
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

    if (tid == 0u) {
        var max_score: f32 = -3.402823e+38;
        for (var k: u32 = start_k; k < kv_len; k++) {
            max_score = max(max_score, shared_scores[k]);
        }
        shared_max = max_score;
    }
    workgroupBarrier();

    let max_score = shared_max;

    if (tid == 0u) {
        var sum_exp: f32 = 0.0;
        for (var k: u32 = start_k; k < kv_len; k++) {
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
        for (var k_pos: u32 = start_k; k_pos < kv_len; k_pos++) {
            let v_idx = get_kv_pos(k_pos);
            let v_offset = v_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
            let v_val = f32(V[v_offset]);
            acc += shared_scores[k_pos] * v_val;
        }
        shared_acc[tid] = acc * inv_sum;
    }
    workgroupBarrier();

    if (valid) {
        let out_offset = head_idx * head_dim + tid;
        output[out_offset] = f16(shared_acc[tid]);
    }
}
