// Streaming Multi-Head Attention Kernel (f16 KV, no workgroup storage)
//
// Same as attention_streaming.wgsl but K/V are stored as f16.

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
    start_pos: u32,  // Absolute position offset for causal masking
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f16>;
@group(0) @binding(3) var<storage, read> V: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn getKVHeadIdx(queryHeadIdx: u32) -> u32 {
    let headsPerKV = u.num_heads / u.num_kv_heads;
    return queryHeadIdx / headsPerKV;
}

fn isMasked(queryPos: u32, keyPos: u32) -> bool {
    if (u.is_causal == 0u) { return false; }
    // Use absolute position (queryPos + start_pos) for correct causal masking during decode
    return keyPos > (queryPos + u.start_pos);
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let linear = wg_id.x;
    let num_heads = u.num_heads;
    let headIdx = linear % num_heads;
    let queryPos = linear / num_heads;

    if (queryPos >= u.query_len) { return; }

    let kvHeadIdx = getKVHeadIdx(headIdx);
    let head_dim = u.head_dim;
    let seq_len = u.seq_len;
    let scale = u.scale;

    var q_local: array<f32, 256>;
    let q_offset = queryPos * num_heads * head_dim + headIdx * head_dim;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        q_local[d] = Q[q_offset + d];
    }

    var maxScore: f32 = -3.402823e+38;
    for (var kPos: u32 = 0u; kPos < seq_len; kPos = kPos + 1u) {
        if (isMasked(queryPos, kPos)) { continue; }
        let k_offset = kPos * u.num_kv_heads * head_dim + kvHeadIdx * head_dim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            dot = dot + q_local[d] * f32(K[k_offset + d]);
        }
        dot = dot * scale;
        maxScore = max(maxScore, dot);
    }

    var sumExp: f32 = 0.0;
    var acc: array<f32, 256>;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        acc[d] = 0.0;
    }

    for (var kPos: u32 = 0u; kPos < seq_len; kPos = kPos + 1u) {
        if (isMasked(queryPos, kPos)) { continue; }
        let k_offset = kPos * u.num_kv_heads * head_dim + kvHeadIdx * head_dim;
        let v_offset = kPos * u.num_kv_heads * head_dim + kvHeadIdx * head_dim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            dot = dot + q_local[d] * f32(K[k_offset + d]);
        }
        dot = dot * scale;
        let w = exp(dot - maxScore);
        sumExp = sumExp + w;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            acc[d] = acc[d] + w * f32(V[v_offset + d]);
        }
    }

    if (sumExp <= 0.0) { return; }

    let out_offset = queryPos * num_heads * head_dim + headIdx * head_dim;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        output[out_offset + d] = acc[d] / sumExp;
    }
}

