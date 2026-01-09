// Fused Multi-Head Attention Kernel (small tiles)
//
// Compatibility-focused variant for devices with limited workgroup storage
// and for models with larger head_dim (up to MAX_HEAD_DIM).
//
// Differences vs attention.wgsl:
// - BLOCK_SIZE = 32, HEAD_TILE = 32
// - No shared score matrix (recompute in second pass)
// - Tiles over head_dim to support large heads
//
// Q and output are f32. K/V are f32.
//
// One workgroup per (query_block, head) encoded linearly in workgroup_id.x:
//   linear = workgroup_id.x
//   head_idx = linear % num_heads
//   query_block_idx = linear / num_heads
//
// workgroup_size = BLOCK_SIZE (one thread per query position in the block).

const BLOCK_SIZE: u32 = 32u;   // Sequence tile size
const HEAD_TILE: u32 = 32u;    // Head dimension tile
const MAX_HEAD_DIM: u32 = 256u;
override WORKGROUP_SIZE: u32 = 32u;  // Workgroup size

struct Uniforms {
    num_heads: u32,       // Number of query heads
    num_kv_heads: u32,    // Number of KV heads (for GQA)
    head_dim: u32,        // Dimension per head
    seq_len: u32,         // KV length
    query_len: u32,       // Query length (1 for decode, >1 for prefill)
    scale: f32,           // 1/sqrt(head_dim)
    is_causal: u32,       // Apply causal mask (1 = yes)
    start_pos: u32,       // Absolute position offset for causal masking
    attn_softcap: f32,    // Gemma 2: 50.0, 0 = disabled
    sliding_window: u32,  // Sliding window size (0 = disabled, >0 = window size)
    kv_len_source: u32,   // 0 = use uniform seq_len, 1 = use buffer
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;       // [query_len, num_heads, head_dim]
@group(0) @binding(2) var<storage, read> K: array<f32>;       // [seq_len, num_kv_heads, head_dim]
@group(0) @binding(3) var<storage, read> V: array<f32>;       // [seq_len, num_kv_heads, head_dim]
@group(0) @binding(4) var<storage, read_write> output: array<f32>; // [query_len, num_heads, head_dim]
@group(0) @binding(5) var<storage, read> kv_len_buffer: array<u32>;

// Shared memory for tiled K/V slices
var<workgroup> shared_K: array<f32, BLOCK_SIZE * HEAD_TILE>;
var<workgroup> shared_V: array<f32, BLOCK_SIZE * HEAD_TILE>;

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

fn is_masked(query_pos: u32, key_pos: u32) -> bool {
    let abs_query = query_pos + u.start_pos;
    let abs_key = key_pos;
    // Causal mask
    if (u.is_causal != 0u && abs_key > abs_query) { return true; }
    // Sliding window mask
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
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let linear = wg_id.x;
    let num_heads = u.num_heads;
    let head_idx = linear % num_heads;
    let query_block_idx = linear / num_heads;
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

    var q_local: array<f32, MAX_HEAD_DIM>;
    var acc: array<f32, MAX_HEAD_DIM>;

    if (valid_query) {
        let q_offset = query_pos * num_heads * head_dim + head_idx * head_dim;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            q_local[d] = Q[q_offset + d];
            acc[d] = 0.0;
        }
    }

    var m_i: f32 = -3.402823e+38;
    var l_i: f32 = 0.0;

    let num_kv_blocks = (seq_len + BLOCK_SIZE - 1u) / BLOCK_SIZE;
    let num_head_tiles = (head_dim + HEAD_TILE - 1u) / HEAD_TILE;

    for (var kv_block: u32 = 0u; kv_block < num_kv_blocks; kv_block = kv_block + 1u) {
        let kv_block_start = kv_block * BLOCK_SIZE;

        var scores: array<f32, BLOCK_SIZE>;
        for (var k_init: u32 = 0u; k_init < BLOCK_SIZE; k_init = k_init + 1u) {
            scores[k_init] = 0.0;
        }

        // Compute dot products for this KV block by tiling head_dim.
        for (var ht: u32 = 0u; ht < num_head_tiles; ht = ht + 1u) {
            let d0 = ht * HEAD_TILE;
            let tile_len = min(HEAD_TILE, head_dim - d0);

            // Load K slice for this block into shared memory.
            let key_pos_load = kv_block_start + thread_idx;
            if (key_pos_load < seq_len) {
                let k_offset = key_pos_load * u.num_kv_heads * head_dim + kv_head_idx * head_dim + d0;
                for (var td: u32 = 0u; td < tile_len; td = td + 1u) {
                    shared_K[thread_idx * HEAD_TILE + td] = K[k_offset + td];
                }
            } else {
                for (var td: u32 = 0u; td < tile_len; td = td + 1u) {
                    shared_K[thread_idx * HEAD_TILE + td] = 0.0;
                }
            }

            workgroupBarrier();

            if (valid_query) {
                for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                    let key_pos = kv_block_start + k;
                    if (key_pos >= seq_len) { continue; }
                    if (is_masked(query_pos, key_pos)) { continue; }

                    var dot_partial: f32 = 0.0;
                    for (var td: u32 = 0u; td < tile_len; td = td + 1u) {
                        dot_partial = dot_partial + q_local[d0 + td] * shared_K[k * HEAD_TILE + td];
                    }
                    scores[k] = scores[k] + dot_partial;
                }
            }

            workgroupBarrier();
        }

        var m_new: f32 = m_i;
        if (valid_query) {
            var block_max: f32 = -3.402823e+38;
            for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                let key_pos = kv_block_start + k;
                if (key_pos >= seq_len) { continue; }
                if (is_masked(query_pos, key_pos)) { continue; }

                var s = scores[k] * scale;
                // Gemma 2 attention softcapping
                if (u.attn_softcap > 0.0) {
                    s = tanh(s / u.attn_softcap) * u.attn_softcap;
                }
                scores[k] = s;
                block_max = max(block_max, s);
            }

            m_new = max(m_i, block_max);
            let correction = exp(m_i - m_new);

            l_i = l_i * correction;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                acc[d] = acc[d] * correction;
            }
        }

        // Accumulate V contribution by tiling head_dim again.
        // Barriers must be uniform, so only math is guarded by valid_query.
        for (var ht: u32 = 0u; ht < num_head_tiles; ht = ht + 1u) {
            let d0 = ht * HEAD_TILE;
            let tile_len = min(HEAD_TILE, head_dim - d0);

            let key_pos_load = kv_block_start + thread_idx;
            if (key_pos_load < seq_len) {
                let v_offset = key_pos_load * u.num_kv_heads * head_dim + kv_head_idx * head_dim + d0;
                for (var td: u32 = 0u; td < tile_len; td = td + 1u) {
                    shared_V[thread_idx * HEAD_TILE + td] = V[v_offset + td];
                }
            } else {
                for (var td: u32 = 0u; td < tile_len; td = td + 1u) {
                    shared_V[thread_idx * HEAD_TILE + td] = 0.0;
                }
            }

            workgroupBarrier();

            if (valid_query) {
                for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                    let key_pos = kv_block_start + k;
                    if (key_pos >= seq_len) { continue; }
                    if (is_masked(query_pos, key_pos)) { continue; }

                    let p = exp(scores[k] - m_new);
                    // Only accumulate l_i on first head tile to avoid double counting
                    if (ht == 0u) {
                        l_i = l_i + p;
                    }

                    for (var td: u32 = 0u; td < tile_len; td = td + 1u) {
                        acc[d0 + td] = acc[d0 + td] + p * shared_V[k * HEAD_TILE + td];
                    }
                }
            }

            workgroupBarrier();
        }

        if (valid_query) {
            m_i = m_new;
        }
    }

    // Always write output when valid_query - write zeros if l_i is 0 to avoid garbage
    if (valid_query) {
        let out_offset = query_pos * num_heads * head_dim + head_idx * head_dim;
        let inv_l_i = select(0.0, 1.0 / l_i, l_i > 0.0);
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            output[out_offset + d] = acc[d] * inv_l_i;
        }
    }
}
