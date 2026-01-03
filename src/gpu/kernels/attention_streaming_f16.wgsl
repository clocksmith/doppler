// Streaming Multi-Head Attention Kernel (f16 QKV + f16 output)
//
// Same as attention_streaming.wgsl but Q/K/V are f16 and output is f16.

enable f16;

override WORKGROUP_SIZE: u32 = 1u;
const MAX_HEAD_DIM: u32 = 256u;

struct Uniforms {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    query_len: u32,
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

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

fn is_masked(query_pos: u32, key_pos: u32) -> bool {
    let abs_query = query_pos + u.start_pos;
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
        return u.seq_len;
    }
    return kv_len_buffer[0];
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let linear = wg_id.x;
    let num_heads = u.num_heads;
    let head_idx = linear % num_heads;
    let query_pos = linear / num_heads;

    if (query_pos >= u.query_len) { return; }

    let kv_head_idx = get_kv_head_idx(head_idx);
    let head_dim = u.head_dim;
    if (head_dim > MAX_HEAD_DIM) { return; }
    let seq_len = get_kv_len();
    let scale = u.scale;

    var q_local: array<f32, MAX_HEAD_DIM>;
    let q_offset = query_pos * num_heads * head_dim + head_idx * head_dim;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        q_local[d] = f32(Q[q_offset + d]);
    }

    var max_score: f32 = -3.402823e+38;
    for (var k_pos: u32 = 0u; k_pos < seq_len; k_pos = k_pos + 1u) {
        if (is_masked(query_pos, k_pos)) { continue; }
        let k_idx = get_kv_pos(k_pos);
        let k_offset = k_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            dot = dot + q_local[d] * f32(K[k_offset + d]);
        }
        dot = dot * scale;
        if (u.attn_softcap > 0.0) {
            dot = tanh(dot / u.attn_softcap) * u.attn_softcap;
        }
        max_score = max(max_score, dot);
    }

    var sum_exp: f32 = 0.0;
    var acc: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        acc[d] = 0.0;
    }

    for (var k_pos: u32 = 0u; k_pos < seq_len; k_pos = k_pos + 1u) {
        if (is_masked(query_pos, k_pos)) { continue; }
        let k_idx = get_kv_pos(k_pos);
        let k_offset = k_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim;
        let v_offset = k_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            dot = dot + q_local[d] * f32(K[k_offset + d]);
        }
        dot = dot * scale;
        if (u.attn_softcap > 0.0) {
            dot = tanh(dot / u.attn_softcap) * u.attn_softcap;
        }
        let w = exp(dot - max_score);
        sum_exp = sum_exp + w;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            acc[d] = acc[d] + w * f32(V[v_offset + d]);
        }
    }

    let out_offset = query_pos * num_heads * head_dim + head_idx * head_dim;
    if (sum_exp <= 0.0) {
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            output[out_offset + d] = f16(0.0);
        }
        return;
    }
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        output[out_offset + d] = f16(acc[d] / sum_exp);
    }
}
