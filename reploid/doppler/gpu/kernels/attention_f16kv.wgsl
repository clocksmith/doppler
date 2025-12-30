// Fused Multi-Head Attention Kernel (f16 KV)
//
// Same as attention.wgsl but K/V are stored as f16 for KV-cache compression.
// Q and output remain f32; K/V are cast to f32 on load.

enable f16;

// Tile sizes for blocked attention
override BLOCK_SIZE: u32 = 64u;  // Sequence tile size
override HEAD_TILE: u32 = 64u;   // Head dimension tile
override WORKGROUP_SIZE: u32 = 64u;  // Main kernel workgroup size
override DECODE_WORKGROUP_SIZE: u32 = 256u;  // Decode kernel workgroup size

struct Uniforms {
    num_heads: u32,       // Number of query heads
    num_kv_heads: u32,    // Number of KV heads (for GQA)
    head_dim: u32,        // Dimension per head
    seq_len: u32,         // Current sequence length (for KV)
    query_len: u32,       // Query length (1 for decode, seq_len for prefill)
    scale: f32,           // 1/sqrt(head_dim)
    is_causal: u32,       // Apply causal mask (1 = yes)
    start_pos: u32,       // Absolute position offset for causal masking
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;       // [query_len, numHeads, head_dim]
@group(0) @binding(2) var<storage, read> K: array<f16>;       // [seq_len, numKVHeads, head_dim]
@group(0) @binding(3) var<storage, read> V: array<f16>;       // [seq_len, numKVHeads, head_dim]
@group(0) @binding(4) var<storage, read_write> output: array<f32>; // [query_len, numHeads, head_dim]

// Shared memory for tiled computation
var<workgroup> shared_K: array<f32, 4096>;  // BLOCK_SIZE * HEAD_TILE
var<workgroup> shared_V: array<f32, 4096>;  // BLOCK_SIZE * HEAD_TILE
var<workgroup> shared_scores: array<f32, 4096>;  // BLOCK_SIZE * BLOCK_SIZE

// Online softmax accumulators (per-thread)
// Sized for 256 to support attention_decode workgroup size (prefill uses 64)
var<workgroup> row_max: array<f32, 256>;
var<workgroup> row_sum: array<f32, 256>;

// Get KV head index for grouped query attention
fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    // GQA: multiple query heads share one KV head
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

// Check if position should be masked (causal attention)
fn is_masked(query_pos: u32, key_pos: u32) -> bool {
    if (u.is_causal == 0u) {
        return false;
    }
    // For causal attention, query can only attend to keys at same or earlier positions
    return key_pos > (query_pos + u.start_pos);
}

// Main attention kernel - one workgroup per (query_block, head)
// Workgroups are dispatched linearly as: numQueryBlocks * numHeads.
// head_idx and query_block_idx are derived from workgroup_id.x.
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
    let seq_len = u.seq_len;
    let query_len = u.query_len;
    let scale = u.scale;

    // Query position this thread handles
    let query_pos = query_block_idx * BLOCK_SIZE + thread_idx;
    let valid_query = query_pos < query_len;

    // Initialize online softmax accumulators
    var m_i: f32 = -3.402823e+38;  // -inf for max tracking
    var l_i: f32 = 0.0;            // Sum of exp(x - max)
    var acc: array<f32, 64>;       // Accumulator for output [head_dim], assuming head_dim <= 64

    // Initialize accumulator
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        acc[d] = 0.0;
    }

    // Load query for this thread into registers
    var q_local: array<f32, 64>;
    if (valid_query) {
        let q_offset = query_pos * u.num_heads * head_dim + head_idx * head_dim;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            q_local[d] = Q[q_offset + d];
        }
    }

    // Process key-value blocks
    let num_kv_blocks = (seq_len + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    for (var kv_block: u32 = 0u; kv_block < num_kv_blocks; kv_block = kv_block + 1u) {
        let kv_block_start = kv_block * BLOCK_SIZE;

        // Collaborative load of K block into shared memory
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

        // Load V block
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

        // Compute attention scores for this block
        if (valid_query) {
            // Find max in this block (for numerical stability)
            var block_max: f32 = -3.402823e+38;

            for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                let key_pos = kv_block_start + k;
                if (key_pos >= seq_len) { continue; }

                // Check causal mask
                if (is_masked(query_pos, key_pos)) { continue; }

                // Compute Q @ K^T for this position
                var score: f32 = 0.0;
                for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                    score = score + q_local[d] * shared_K[k * head_dim + d];
                }
                score = score * scale;

                block_max = max(block_max, score);
                shared_scores[thread_idx * BLOCK_SIZE + k] = score;
            }

            // Online softmax update
            let m_new = max(m_i, block_max);
            let correction = exp(m_i - m_new);

            // Rescale previous accumulator
            l_i = l_i * correction;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                acc[d] = acc[d] * correction;
            }

            // Add contribution from this block
            for (var k: u32 = 0u; k < BLOCK_SIZE; k = k + 1u) {
                let key_pos = kv_block_start + k;
                if (key_pos >= seq_len) { continue; }
                if (is_masked(query_pos, key_pos)) { continue; }

                let score = shared_scores[thread_idx * BLOCK_SIZE + k];
                let p = exp(score - m_new);
                l_i = l_i + p;

                // Accumulate V contribution
                for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                    acc[d] = acc[d] + p * shared_V[k * head_dim + d];
                }
            }

            m_i = m_new;
        }

        workgroupBarrier();
    }

    // Normalize by sum and write output
    if (valid_query && l_i > 0.0) {
        let out_offset = query_pos * u.num_heads * head_dim + head_idx * head_dim;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            output[out_offset + d] = acc[d] / l_i;
        }
    }
}

// Simplified single-query attention for decode step
// More efficient when query_len == 1
@compute @workgroup_size(DECODE_WORKGROUP_SIZE, 1, 1)
fn attention_decode(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let head_idx = wg_id.x;
    let thread_idx = local_id.x;

    let kv_head_idx = get_kv_head_idx(head_idx);
    let head_dim = u.head_dim;
    let seq_len = u.seq_len;
    let scale = u.scale;

    // Each thread handles a subset of key positions
    let keys_per_thread = (seq_len + 255u) / 256u;

    // Load query (single position)
    var q_local: array<f32, 128>;  // Support up to 128 head_dim
    let q_offset = head_idx * head_dim;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        q_local[d] = Q[q_offset + d];
    }

    // Compute partial attention scores and find local max
    var local_max: f32 = -3.402823e+38;
    var local_scores: array<f32, 32>;  // Store scores for this thread's keys
    var local_count: u32 = 0u;

    for (var i: u32 = 0u; i < keys_per_thread; i = i + 1u) {
        let key_pos = thread_idx * keys_per_thread + i;
        if (key_pos >= seq_len) { break; }

        // Causal: can attend to all previous positions (query is at end)
        let k_offset = key_pos * u.num_kv_heads * head_dim + kv_head_idx * head_dim;

        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            score = score + q_local[d] * f32(K[k_offset + d]);
        }
        score = score * scale;

        local_scores[i] = score;
        local_max = max(local_max, score);
        local_count = local_count + 1u;
    }

    // Store local max for reduction
    row_max[thread_idx] = local_max;
    workgroupBarrier();

    // Parallel reduction to find global max
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride && thread_idx + stride < 256u) {
            row_max[thread_idx] = max(row_max[thread_idx], row_max[thread_idx + stride]);
        }
        workgroupBarrier();
    }

    let global_max = row_max[0];

    // Compute exp(score - max) and local sum
    var local_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < local_count; i = i + 1u) {
        local_scores[i] = exp(local_scores[i] - global_max);
        local_sum = local_sum + local_scores[i];
    }

    row_sum[thread_idx] = local_sum;
    workgroupBarrier();

    // Parallel reduction for sum
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride && thread_idx + stride < 256u) {
            row_sum[thread_idx] = row_sum[thread_idx] + row_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }

    let global_sum = row_sum[0];

    // Compute weighted V contribution
    var local_out: array<f32, 128>;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        local_out[d] = 0.0;
    }

    for (var i: u32 = 0u; i < local_count; i = i + 1u) {
        let key_pos = thread_idx * keys_per_thread + i;
        let v_offset = key_pos * u.num_kv_heads * head_dim + kv_head_idx * head_dim;
        let weight = local_scores[i] / global_sum;

        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            local_out[d] = local_out[d] + weight * f32(V[v_offset + d]);
        }
    }

    // Reduction for output (atomic add or shared memory reduction)
    // For simplicity, use shared memory
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        shared_V[thread_idx * head_dim + d] = local_out[d];
    }
    workgroupBarrier();

    // Thread 0 sums all contributions
    if (thread_idx == 0u) {
        let out_offset = head_idx * head_dim;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            var sum: f32 = 0.0;
            for (var t: u32 = 0u; t < 256u; t = t + 1u) {
                sum = sum + shared_V[t * head_dim + d];
            }
            output[out_offset + d] = sum;
        }
    }
}
