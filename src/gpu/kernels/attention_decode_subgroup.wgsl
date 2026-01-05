// Attention Decode Kernel - Subgroup Optimized
//
// Optimized for seqLen=1 (decode) using subgroup operations.
// Requires headDim <= subgroup_size for correct operation.
//
// Architecture:
// - One workgroup per head
// - workgroup_size = max(headDim, 256)
// - Uses subgroup operations for fast reductions

enable subgroups;

// Workgroup size for decode
override WORKGROUP_SIZE: u32 = 256u;

// Uniforms must match TypeScript createAttentionUniformBuffer() layout exactly:
// offset 0: numHeads, offset 4: numKVHeads, offset 8: headDim,
// offset 12: kvLen, offset 16: seqLen, offset 20: scale, offset 24: causal, offset 28: startPos, offset 40: kvLenSource
struct Uniforms {
    num_heads: u32,       // Number of query heads
    num_kv_heads: u32,    // Number of KV heads (GQA support)
    head_dim: u32,        // Head dimension
    kv_len: u32,          // Current KV cache length
    seq_len: u32,         // Always 1 for decode
    scale: f32,           // Attention scale (1/sqrt(headDim))
    causal: u32,          // Causal masking flag
    start_pos: u32,       // Start position for RoPE
    attn_softcap: f32,    // Gemma 2: 50.0, 0 = disabled
    sliding_window: u32,  // Sliding window size (0 = disabled, >0 = window size)
    kv_len_source: u32,   // 0 = use uniform kv_len, 1 = use buffer
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K_cache: array<f32>;
@group(0) @binding(3) var<storage, read> V_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> kv_len_buffer: array<u32>;

// Shared memory for attention scores and cross-subgroup reduction
var<workgroup> scores: array<f32, 2048>;
var<workgroup> subgroup_sums: array<f32, 8>;  // For 8 subgroups of size 32
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;

// Check if position should be masked (sliding window attention)
// For decode, query_pos = start_pos (we're at the latest position)
fn is_masked(key_pos: u32) -> bool {
    let abs_query = u.start_pos;
    let abs_key = key_pos;
    // Causal mask (key must be <= query position)
    if (u.causal != 0u && abs_key > abs_query) { return true; }
    // Sliding window mask
    if (u.sliding_window > 0u && abs_query >= u.sliding_window) {
        if (abs_key < abs_query - u.sliding_window + 1u) { return true; }
    }
    return false;
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
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_tid: u32,
) {
    let head_idx = workgroup_id.x;
    let tid = local_id.x;
    let head_dim = u.head_dim;
    let kv_len = get_kv_len();
    let valid_thread = tid < head_dim;
    let subgroup_id = tid / subgroup_size;
    let num_subgroups = (head_dim + subgroup_size - 1u) / subgroup_size;

    // GQA: map query head to KV head
    let kv_head_idx = head_idx / (u.num_heads / u.num_kv_heads);

    // Load Q value for this thread
    var q_val = 0.0;
    if (valid_thread) {
        let q_offset = head_idx * head_dim + tid;
        q_val = Q[q_offset];
    }

    let scale = 1.0 / sqrt(f32(head_dim));

    // Phase 1: Compute attention scores (Q @ K^T)
    for (var k = 0u; k < kv_len; k++) {
        var k_val = 0.0;
        if (valid_thread) {
            let k_offset = k * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
            k_val = K_cache[k_offset];
        }

        // Compute partial dot product within subgroup
        let dot = q_val * k_val;
        var partial_sum = subgroupAdd(dot);

        // First thread of each subgroup writes to shared memory
        if (subgroup_tid == 0u && subgroup_id < num_subgroups) {
            subgroup_sums[subgroup_id] = partial_sum;
        }
        workgroupBarrier();

        // Thread 0 sums all subgroup contributions
        if (tid == 0u) {
            var total = 0.0;
            for (var s = 0u; s < num_subgroups; s++) {
                total += subgroup_sums[s];
            }
            var s = total * scale;
            // Gemma 2 attention softcapping
            if (u.attn_softcap > 0.0) {
                s = tanh(s / u.attn_softcap) * u.attn_softcap;
            }
            // Apply masking (causal + sliding window)
            if (is_masked(k)) {
                s = -1e38;
            }
            scores[k] = s;
        }
        workgroupBarrier();
    }

    // Phase 2: Softmax - find max
    var max_score = -1e38;
    if (valid_thread) {
        for (var k = tid; k < kv_len; k += head_dim) {
            max_score = max(max_score, scores[k]);
        }
    }

    // Cross-subgroup max reduction
    var subgroup_max = subgroupMax(max_score);
    if (subgroup_tid == 0u && subgroup_id < num_subgroups) {
        subgroup_sums[subgroup_id] = subgroup_max;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var global_max = -1e38;
        for (var s = 0u; s < num_subgroups; s++) {
            global_max = max(global_max, subgroup_sums[s]);
        }
        shared_max = global_max;
    }
    workgroupBarrier();

    let global_max = shared_max;

    // Compute exp and sum
    var sum_exp = 0.0;
    if (valid_thread) {
        for (var k = tid; k < kv_len; k += head_dim) {
            let exp_val = exp(scores[k] - global_max);
            scores[k] = exp_val;
            sum_exp += exp_val;
        }
    }

    // Cross-subgroup sum reduction
    var subgroup_sum = subgroupAdd(sum_exp);
    if (subgroup_tid == 0u && subgroup_id < num_subgroups) {
        subgroup_sums[subgroup_id] = subgroup_sum;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var global_sum = 0.0;
        for (var s = 0u; s < num_subgroups; s++) {
            global_sum += subgroup_sums[s];
        }
        shared_sum = global_sum;
    }
    workgroupBarrier();

    let global_sum = shared_sum;

    // Normalize
    if (valid_thread) {
        for (var k = tid; k < kv_len; k += head_dim) {
            scores[k] /= global_sum;
        }
    }
    workgroupBarrier();

    // Phase 3: Weighted sum (scores @ V)
    var output_val = 0.0;
    if (valid_thread) {
        for (var k = 0u; k < kv_len; k++) {
            let v_offset = k * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
            let v_val = V_cache[v_offset];
            output_val += scores[k] * v_val;
        }

        // Write output
        let out_offset = head_idx * head_dim + tid;
        output[out_offset] = output_val;
    }
}
