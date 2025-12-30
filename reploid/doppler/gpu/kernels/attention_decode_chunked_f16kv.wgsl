// Chunked Decode Attention Kernel (f16 KV)
//
// Optimized for models with few heads and large headDim (e.g., Gemma 3: 4 heads × 256 dim)
// Uses parallel reduction across headDim within each workgroup.
//
// Dispatch: numHeads workgroups, each with 256 threads
// Each thread handles one dimension of the head.

enable f16;

// Workgroup size for decode
override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,       // Current KV cache length
    seq_len: u32,      // Query length (1 for decode)
    scale: f32,
    is_causal: u32,
    start_pos: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f16>;
@group(0) @binding(3) var<storage, read> V: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Shared memory for parallel reductions
var<workgroup> shared_scores: array<f32, 2048>;  // Attention scores for each KV position
var<workgroup> shared_partial: array<f32, 256>;   // Partial sums for reduction
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;
var<workgroup> shared_acc: array<f32, 256>;       // Accumulated output

fn get_kv_head_idx(query_head_idx: u32) -> u32 {
    let heads_per_kv = u.num_heads / u.num_kv_heads;
    return query_head_idx / heads_per_kv;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let head_idx = workgroup_id.x;
    let tid = local_id.x;
    let head_dim = u.head_dim;
    let kv_len = u.kv_len;
    let scale = u.scale;
    let kv_head_idx = get_kv_head_idx(head_idx);

    // Only threads < headDim participate in computation
    let valid = tid < head_dim;

    // Load Q value for this thread's dimension
    var q_val: f32 = 0.0;
    if (valid) {
        let q_offset = head_idx * head_dim + tid;
        q_val = Q[q_offset];
    }

    // Initialize accumulator
    if (valid) {
        shared_acc[tid] = 0.0;
    }

    // Phase 1: Compute attention scores (Q @ K^T) in parallel
    // Each thread computes partial dot product for its dimension
    for (var kPos: u32 = 0u; k_pos < kv_len; kPos++) {
        var k_val: f32 = 0.0;
        if (valid) {
            let k_offset = k_pos * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
            k_val = f32(K[k_offset]);
        }

        // Partial dot product
        let partial = q_val * k_val;
        shared_partial[tid] = partial;
        workgroupBarrier();

        // Parallel reduction for dot product (thread 0 accumulates)
        // Using sequential reduction for correctness - could optimize with tree reduction
        if (tid == 0u) {
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d++) {
                dot += shared_partial[d];
            }
            shared_scores[k_pos] = dot * scale;
        }
        workgroupBarrier();
    }

    // Phase 2: Find max score for numerical stability
    if (tid == 0u) {
        var max_score: f32 = -3.402823e+38;
        for (var k: u32 = 0u; k < kvLen; k++) {
            max_score = max(max_score, shared_scores[k]);
        }
        shared_max = max_score;
    }
    workgroupBarrier();

    let max_score = shared_max;

    // Phase 3: Compute softmax weights and accumulate V
    if (tid == 0u) {
        var sum_exp: f32 = 0.0;
        for (var k: u32 = 0u; k < kvLen; k++) {
            let w = exp(shared_scores[k] - max_score);
            shared_scores[k] = w;  // Store weight for V accumulation
            sum_exp += w;
        }
        shared_sum = sum_exp;
    }
    workgroupBarrier();

    let sum_exp = shared_sum;
    // Use select instead of early return to maintain uniform control flow
    let inv_sum = select(0.0, 1.0 / sum_exp, sum_exp > 0.0);

    // Phase 4: Weighted sum of V (parallel across dimensions)
    if (valid) {
        var acc: f32 = 0.0;
        for (var kPos: u32 = 0u; k_pos < kv_len; kPos++) {
            let v_offset = k_pos * u.num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
            let v_val = f32(V[v_offset]);
            acc += shared_scores[k_pos] * v_val;
        }
        shared_acc[tid] = acc * inv_sum;
    }
    workgroupBarrier();

    // Phase 5: Write output
    if (valid) {
        let out_offset = head_idx * head_dim + tid;
        output[out_offset] = shared_acc[tid];
    }
}
