// Streaming Multi-Head Attention Kernel (no workgroup storage)
//
// Fallback variant for devices with extremely small shared memory or
// models with head_dim beyond tiled support. Uses two-pass softmax and
// reads K/V directly from storage. Slower but compatible.

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
    start_pos: u32,  // Absolute position offset for causal masking
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

fn is_masked(query_pos: u32, key_pos: u32) -> bool {
    if (u.is_causal == 0u) { return false; }
    return key_pos > (query_pos + u.start_pos);
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
    let seq_len = u.seq_len;
    let scale = u.scale;

    var q_local: array<f32, 256>;
    let q_offset = query_pos * num_heads * head_dim + head_idx * head_dim;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        q_local[d] = Q[q_offset + d];
    }

    var max_score: f32 = -3.402823e+38;
    for (var k_pos: u32 = 0u; k_pos < seq_len; k_pos = k_pos + 1u) {
        if (is_masked(query_pos, k_pos)) { continue; }
        let k_offset = k_pos * u.num_kv_heads * head_dim + kv_head_idx * head_dim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            dot = dot + q_local[d] * K[k_offset + d];
        }
        dot = dot * scale;
        max_score = max(max_score, dot);
    }

    var sum_exp: f32 = 0.0;
    var acc: array<f32, 256>;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        acc[d] = 0.0;
    }

    for (var k_pos: u32 = 0u; k_pos < seq_len; k_pos = k_pos + 1u) {
        if (is_masked(query_pos, k_pos)) { continue; }
        let k_offset = k_pos * u.num_kv_heads * head_dim + kv_head_idx * head_dim;
        let v_offset = k_pos * u.num_kv_heads * head_dim + kv_head_idx * head_dim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            dot = dot + q_local[d] * K[k_offset + d];
        }
        dot = dot * scale;
        let w = exp(dot - max_score);
        sum_exp = sum_exp + w;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            acc[d] = acc[d] + w * V[v_offset + d];
        }
    }

    if (sum_exp <= 0.0) { return; }

    let out_offset = query_pos * num_heads * head_dim + head_idx * head_dim;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        output[out_offset + d] = acc[d] / sum_exp;
    }
}

