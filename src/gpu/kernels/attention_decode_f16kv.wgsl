// Fused Multi-Head Attention Kernel (f16 KV)
//
// Same as attention.wgsl but K/V are stored as f16 for KV-cache compression.
// Q and output remain f32; K/V are cast to f32 on load.

enable f16;

const MAX_HEAD_DIM: u32 = 64u;
const MAX_WORKGROUP_SIZE: u32 = 256u;

override DECODE_WORKGROUP_SIZE: u32 = 256u;  // Decode kernel workgroup size (must match MAX_WORKGROUP_SIZE)

struct Uniforms {
    num_heads: u32,       // Number of query heads
    num_kv_heads: u32,    // Number of KV heads (for GQA)
    head_dim: u32,        // Dimension per head
    seq_len: u32,         // Current sequence length (for KV)
    query_len: u32,       // Query length (1 for decode, seq_len for prefill)
    scale: f32,           // 1/sqrt(head_dim)
    is_causal: u32,       // Apply causal mask (1 = yes)
    start_pos: u32,       // Absolute position offset for causal masking
    attn_softcap: f32,    // Gemma 2: 50.0, 0 = disabled
    sliding_window: u32,  // Sliding window size (0 = disabled, >0 = window size)
    kv_len_source: u32,   // 0 = use uniform seq_len, 1 = use buffer
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;       // [query_len, num_heads, head_dim]
@group(0) @binding(2) var<storage, read> K: array<f16>;       // [seq_len, num_kv_heads, head_dim]
@group(0) @binding(3) var<storage, read> V: array<f16>;       // [seq_len, num_kv_heads, head_dim]
@group(0) @binding(4) var<storage, read_write> output: array<f32>; // [query_len, num_heads, head_dim]
@group(0) @binding(5) var<storage, read> kv_len_buffer: array<u32>;

// Online softmax accumulators (per-thread)
// Sized for 256 to support attention_decode workgroup size
var<workgroup> row_max: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> row_sum: array<f32, MAX_WORKGROUP_SIZE>;

// Get KV head index for grouped query attention
fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    // GQA: multiple query heads share one KV head
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

// Check if position should be masked (causal + sliding window attention)
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

// Main attention kernel - one workgroup per (query_block, head)
// Workgroups are dispatched linearly as: numQueryBlocks * numHeads.
// head_idx and query_block_idx are derived from workgroup_id.x.

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

    if (DECODE_WORKGROUP_SIZE > MAX_WORKGROUP_SIZE) {
        return;
    }

    let kv_head_idx = get_kv_head_idx(head_idx);
    let head_dim = u.head_dim;
    if (head_dim > MAX_HEAD_DIM) {
        return;
    }
    let seq_len = get_kv_len();
    let scale = u.scale;

    // Each thread handles a subset of key positions
    let keys_per_thread = (seq_len + DECODE_WORKGROUP_SIZE - 1u) / DECODE_WORKGROUP_SIZE;

    // Load query (single position)
    var q_local: array<f32, MAX_HEAD_DIM>;
    let q_offset = head_idx * head_dim;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        q_local[d] = Q[q_offset + d];
    }

    // Pass 1: local max
    var local_max: f32 = -3.402823e+38;
    let query_pos = 0u;

    for (var i: u32 = 0u; i < keys_per_thread; i = i + 1u) {
        let key_pos = thread_idx * keys_per_thread + i;
        if (key_pos >= seq_len) { break; }
        if (is_masked(query_pos, key_pos)) { continue; }

        let k_offset = key_pos * u.num_kv_heads * head_dim + kv_head_idx * head_dim;

        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            score = score + q_local[d] * f32(K[k_offset + d]);
        }
        score = score * scale;

        if (u.attn_softcap > 0.0) {
            score = tanh(score / u.attn_softcap) * u.attn_softcap;
        }

        local_max = max(local_max, score);
    }

    // Store local max for reduction
    row_max[thread_idx] = local_max;
    workgroupBarrier();

    // Parallel reduction to find global max
    for (var stride: u32 = DECODE_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride && thread_idx + stride < DECODE_WORKGROUP_SIZE) {
            row_max[thread_idx] = max(row_max[thread_idx], row_max[thread_idx + stride]);
        }
        workgroupBarrier();
    }

    let global_max = row_max[0];

    // Pass 2: local sum and output accumulator
    var local_sum: f32 = 0.0;
    var local_out: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        local_out[d] = 0.0;
    }

    for (var i: u32 = 0u; i < keys_per_thread; i = i + 1u) {
        let key_pos = thread_idx * keys_per_thread + i;
        if (key_pos >= seq_len) { break; }
        if (is_masked(query_pos, key_pos)) { continue; }

        let k_offset = key_pos * u.num_kv_heads * head_dim + kv_head_idx * head_dim;
        let v_offset = key_pos * u.num_kv_heads * head_dim + kv_head_idx * head_dim;

        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            score = score + q_local[d] * f32(K[k_offset + d]);
        }
        score = score * scale;

        if (u.attn_softcap > 0.0) {
            score = tanh(score / u.attn_softcap) * u.attn_softcap;
        }

        let exp_score = exp(score - global_max);
        local_sum = local_sum + exp_score;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            local_out[d] = local_out[d] + exp_score * f32(V[v_offset + d]);
        }
    }

    row_sum[thread_idx] = local_sum;
    workgroupBarrier();

    // Parallel reduction for sum
    for (var stride: u32 = DECODE_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride && thread_idx + stride < DECODE_WORKGROUP_SIZE) {
            row_sum[thread_idx] = row_sum[thread_idx] + row_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }

    let global_sum = row_sum[0];
    let inv_sum = select(0.0, 1.0 / global_sum, global_sum > 0.0);

    // Reduction for output (per-dimension)
    let out_offset = head_idx * head_dim;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        row_sum[thread_idx] = local_out[d];
        workgroupBarrier();
        if (thread_idx == 0u) {
            var sum: f32 = 0.0;
            for (var t: u32 = 0u; t < DECODE_WORKGROUP_SIZE; t = t + 1u) {
                sum = sum + row_sum[t];
            }
            output[out_offset + d] = sum * inv_sum;
        }
        workgroupBarrier();
    }
}
