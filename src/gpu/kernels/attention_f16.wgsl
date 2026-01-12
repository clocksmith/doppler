// Fused Multi-Head Attention Kernel (f16 QKV + f16 output)
//
// Same algorithm as attention.wgsl but Q/K/V are f16 and output is f16.
// Computation is performed in f32 for stability.

enable f16;

// Tile sizes for blocked attention
const MAX_BLOCK_SIZE: u32 = 64u;
const MAX_HEAD_TILE: u32 = 64u;
const MAX_HEAD_DIM: u32 = 64u;

override BLOCK_SIZE: u32 = 64u;
override HEAD_TILE: u32 = 64u;
override WORKGROUP_SIZE: u32 = 64u;

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
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f16>;
@group(0) @binding(2) var<storage, read> K: array<f16>;
@group(0) @binding(3) var<storage, read> V: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f16>;
@group(0) @binding(5) var<storage, read> kv_len_buffer: array<u32>;

var<workgroup> shared_K: array<f32, MAX_BLOCK_SIZE * MAX_HEAD_TILE>;
var<workgroup> shared_V: array<f32, MAX_BLOCK_SIZE * MAX_HEAD_TILE>;
var<workgroup> shared_scores: array<f32, MAX_BLOCK_SIZE * MAX_BLOCK_SIZE>;

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

fn is_masked(query_pos: u32, key_pos: u32) -> bool {
    let abs_query = query_pos + u.start_pos;
    let abs_key = key_pos;
    if (u.is_causal != 0u && abs_key > abs_query) { return true; }
    if (u.sliding_window > 0u && abs_query >= u.sliding_window) {
        if (abs_key < abs_query - u.sliding_window + 1u) { return true; }
    }
    return false;
}

fn get_kv_len() -> u32 {
    if (u.kv_len_source == 0u) {
        return u.seq_len;
    }
    return kv_len_buffer[0];
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let linear = wg_id.x;
    let head_idx = linear % u.num_heads;
    let query_block_idx = linear / u.num_heads;
    let thread_idx = local_id.x;

    let kv_head_idx = get_kv_head_idx(head_idx);
    let head_dim = u.head_dim;
    if (head_dim > MAX_HEAD_DIM) {
        return;
    }
    let seq_len = get_kv_len();
    let query_len = u.query_len;
    let scale = u.scale;

    let query_pos = query_block_idx * BLOCK_SIZE + thread_idx;
    let valid_query = query_pos < query_len;

    var m_i: f32 = -3.402823e+38;
    var l_i: f32 = 0.0;
    var acc: array<f32, MAX_HEAD_DIM>;

    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        acc[d] = 0.0;
    }

    var q_local: array<f32, MAX_HEAD_DIM>;
    if (valid_query) {
        let q_offset = query_pos * u.num_heads * head_dim + head_idx * head_dim;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            q_local[d] = f32(Q[q_offset + d]);
        }
    }

    let num_kv_blocks = (seq_len + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    for (var kv_block: u32 = 0u; kv_block < num_kv_blocks; kv_block = kv_block + 1u) {
        let kv_block_start = kv_block * BLOCK_SIZE;

        let k_load_idx = kv_block_start + thread_idx;
        if (k_load_idx < seq_len) {
            let k_offset = k_load_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                shared_K[thread_idx * head_dim + d] = f32(K[k_offset + d]);
            }
        } else {
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                shared_K[thread_idx * head_dim + d] = 0.0;
            }
        }

        let v_load_idx = kv_block_start + thread_idx;
        if (v_load_idx < seq_len) {
            let v_offset = v_load_idx * u.num_kv_heads * head_dim + kv_head_idx * head_dim;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                shared_V[thread_idx * head_dim + d] = f32(V[v_offset + d]);
            }
        } else {
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                shared_V[thread_idx * head_dim + d] = 0.0;
            }
        }

        workgroupBarrier();

        if (valid_query) {
            var block_max: f32 = -3.402823e+38;

            for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                let key_pos = kv_block_start + k;
                if (key_pos >= seq_len) { continue; }
                if (is_masked(query_pos, key_pos)) { continue; }

                var score: f32 = 0.0;
                for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                    score = score + q_local[d] * shared_K[k * head_dim + d];
                }
                score = score * scale;

                if (u.attn_softcap > 0.0) {
                    score = tanh(score / u.attn_softcap) * u.attn_softcap;
                }

                block_max = max(block_max, score);
                shared_scores[thread_idx * BLOCK_SIZE + k] = score;
            }

            let m_new = max(m_i, block_max);
            let correction = exp(m_i - m_new);

            l_i = l_i * correction;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                acc[d] = acc[d] * correction;
            }

            for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                let key_pos = kv_block_start + k;
                if (key_pos >= seq_len) { continue; }
                if (is_masked(query_pos, key_pos)) { continue; }

                let score = shared_scores[thread_idx * BLOCK_SIZE + k];
                let p = exp(score - m_new);
                l_i = l_i + p;

                for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                    acc[d] = acc[d] + p * shared_V[k * head_dim + d];
                }
            }

            m_i = m_new;
        }

        workgroupBarrier();
    }

    if (valid_query) {
        let out_offset = query_pos * u.num_heads * head_dim + head_idx * head_dim;
        let inv_l_i = select(0.0, 1.0 / l_i, l_i > 0.0);
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            output[out_offset + d] = f16(acc[d] * inv_l_i);
        }
    }
}
