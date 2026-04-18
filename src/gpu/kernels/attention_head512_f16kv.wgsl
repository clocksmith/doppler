// Fused Multi-Head Attention Kernel (head_dim 512, f16 KV)
//
// Fixed-shape prefill kernel for 512-dim heads. Mirrors attention_head256_f16kv.wgsl
// but doubled head_dim for Gemma-4 global attention layers (layerPattern every_n
// period=5 offset=4 at indices 4/9/14/19/24/29/34). BLOCK_SIZE reduced from 32 to
// 16 to keep shared_block within the 16 KB default workgroup storage limit
// (16 * 128 * 8 = 16 KB vs 32 * 128 * 8 = 32 KB which may exceed the limit).

enable f16;

const BLOCK_SIZE: u32 = 16u;
const WORKGROUP_SIZE: u32 = BLOCK_SIZE;
const HEAD_DIM: u32 = 512u;
const HEAD_DIM_VECS: u32 = 128u;

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
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f16>;
@group(0) @binding(3) var<storage, read> V: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> kv_len_buffer: array<u32>;
@group(0) @binding(6) var<storage, read> page_table: array<u32>;

var<workgroup> shared_block: array<vec4<f16>, BLOCK_SIZE * HEAD_DIM_VECS>;

fn zero_vec4_f16() -> vec4<f16> {
    return vec4<f16>(f16(0.0), f16(0.0), f16(0.0), f16(0.0));
}

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
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
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    if (u.head_dim != HEAD_DIM) {
        return;
    }

    let linear = wg_id.x;
    let num_heads = u.num_heads;
    let head_idx = linear % num_heads;
    let query_block_idx = linear / num_heads;
    let thread_idx = local_id.x;

    let kv_head_idx = get_kv_head_idx(head_idx);
    let seq_len = get_kv_len();
    let query_len = u.query_len;
    let scale = u.scale;

    let query_pos = query_block_idx * BLOCK_SIZE + thread_idx;
    let valid_query = query_pos < query_len;
    let abs_query = query_pos + u.start_pos;

    var q_local: array<vec4<f32>, HEAD_DIM_VECS>;
    var acc: array<vec4<f32>, HEAD_DIM_VECS>;

    if (valid_query) {
        let q_offset = query_pos * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
        for (var d4: u32 = 0u; d4 < HEAD_DIM_VECS; d4 = d4 + 1u) {
            let base = q_offset + d4 * 4u;
            q_local[d4] = vec4<f32>(
                Q[base],
                Q[base + 1u],
                Q[base + 2u],
                Q[base + 3u]
            );
            acc[d4] = vec4<f32>(0.0);
        }
    }

    var m_i: f32 = -3.402823e+38;
    var l_i: f32 = 0.0;

    let num_kv_blocks = (seq_len + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    var min_key_pos: u32 = 0u;
    var max_key_pos: u32 = seq_len;

    if (valid_query) {
        if (u.is_causal != 0u) {
            if (abs_query < u.kv_start) {
                max_key_pos = 0u;
            } else {
                let causal_limit = abs_query - u.kv_start + 1u;
                max_key_pos = min(seq_len, causal_limit);
            }
        }
        if (u.sliding_window > 0u && abs_query >= u.sliding_window) {
            let min_abs_key = abs_query - u.sliding_window + 1u;
            if (min_abs_key > u.kv_start) {
                min_key_pos = min_abs_key - u.kv_start;
            }
            min_key_pos = min(min_key_pos, seq_len);
        }
    }

    for (var kv_block: u32 = 0u; kv_block < num_kv_blocks; kv_block = kv_block + 1u) {
        let kv_block_start = kv_block * BLOCK_SIZE;

        var scores: array<f32, BLOCK_SIZE>;
        var key_active: array<u32, BLOCK_SIZE>;
        var probs: array<f32, BLOCK_SIZE>;
        for (var k_init: u32 = 0u; k_init < BLOCK_SIZE; k_init = k_init + 1u) {
            scores[k_init] = 0.0;
            key_active[k_init] = 0u;
            probs[k_init] = 0.0;
        }

        if (valid_query) {
            for (var k_mask: u32 = 0u; k_mask < BLOCK_SIZE; k_mask = k_mask + 1u) {
                let key_pos = kv_block_start + k_mask;
                if (key_pos < seq_len && key_pos >= min_key_pos && key_pos < max_key_pos) {
                    key_active[k_mask] = 1u;
                }
            }
        }

        let key_pos_load = kv_block_start + thread_idx;
        let shared_row = thread_idx * HEAD_DIM_VECS;
        if (key_pos_load < seq_len) {
            let k_idx = get_kv_pos(key_pos_load);
            let k_offset = k_idx * u.num_kv_heads * HEAD_DIM + kv_head_idx * HEAD_DIM;
            for (var d4: u32 = 0u; d4 < HEAD_DIM_VECS; d4 = d4 + 1u) {
                let base = k_offset + d4 * 4u;
                shared_block[shared_row + d4] = vec4<f16>(
                    K[base],
                    K[base + 1u],
                    K[base + 2u],
                    K[base + 3u]
                );
            }
        } else {
            for (var d4: u32 = 0u; d4 < HEAD_DIM_VECS; d4 = d4 + 1u) {
                shared_block[shared_row + d4] = zero_vec4_f16();
            }
        }

        workgroupBarrier();

        var m_new: f32 = m_i;
        if (valid_query) {
            var block_max: f32 = -3.402823e+38;
            for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                if (key_active[k] == 0u) { continue; }

                let key_row = k * HEAD_DIM_VECS;
                var dot_partial: f32 = 0.0;
                for (var d4: u32 = 0u; d4 < HEAD_DIM_VECS; d4 = d4 + 1u) {
                    dot_partial = dot_partial + dot(
                        q_local[d4],
                        vec4<f32>(shared_block[key_row + d4])
                    );
                }

                var s = dot_partial * scale;
                if (u.attn_softcap > 0.0) {
                    s = tanh(s / u.attn_softcap) * u.attn_softcap;
                }
                scores[k] = s;
                block_max = max(block_max, s);
            }

            m_new = max(m_i, block_max);
            let correction = exp(m_i - m_new);
            l_i = l_i * correction;
            for (var d4: u32 = 0u; d4 < HEAD_DIM_VECS; d4 = d4 + 1u) {
                acc[d4] = acc[d4] * correction;
            }

            for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                if (key_active[k] == 0u) { continue; }
                let p = exp(scores[k] - m_new);
                probs[k] = p;
                l_i = l_i + p;
            }
        }

        workgroupBarrier();

        if (key_pos_load < seq_len) {
            let v_idx = get_kv_pos(key_pos_load);
            let v_offset = v_idx * u.num_kv_heads * HEAD_DIM + kv_head_idx * HEAD_DIM;
            for (var d4: u32 = 0u; d4 < HEAD_DIM_VECS; d4 = d4 + 1u) {
                let base = v_offset + d4 * 4u;
                shared_block[shared_row + d4] = vec4<f16>(
                    V[base],
                    V[base + 1u],
                    V[base + 2u],
                    V[base + 3u]
                );
            }
        } else {
            for (var d4: u32 = 0u; d4 < HEAD_DIM_VECS; d4 = d4 + 1u) {
                shared_block[shared_row + d4] = zero_vec4_f16();
            }
        }

        workgroupBarrier();

        if (valid_query) {
            for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                let p = probs[k];
                if (p == 0.0) { continue; }

                let value_row = k * HEAD_DIM_VECS;
                for (var d4: u32 = 0u; d4 < HEAD_DIM_VECS; d4 = d4 + 1u) {
                    acc[d4] = acc[d4] + p * vec4<f32>(shared_block[value_row + d4]);
                }
            }
            m_i = m_new;
        }

        workgroupBarrier();
    }

    if (valid_query) {
        let out_offset = query_pos * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
        let inv_l_i = select(0.0, 1.0 / l_i, l_i > 0.0);
        for (var d4: u32 = 0u; d4 < HEAD_DIM_VECS; d4 = d4 + 1u) {
            let out_vec = acc[d4] * inv_l_i;
            let base = out_offset + d4 * 4u;
            output[base] = out_vec.x;
            output[base + 1u] = out_vec.y;
            output[base + 2u] = out_vec.z;
            output[base + 3u] = out_vec.w;
        }
    }
}
