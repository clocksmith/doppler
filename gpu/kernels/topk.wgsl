/**
 * Top-K Selection Kernel for MoE Routing
 *
 * Selects top-k experts for each token based on router logits.
 * Optimized for small k (typically 2) and small n (typically 8 experts).
 *
 * Input: softmax probabilities [numTokens, numExperts]
 * Output:
 *   - indices [numTokens, topK] (u32)
 *   - weights [numTokens, topK] (f32, renormalized)
 */

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_tokens: u32,     // Number of tokens
    num_experts: u32,    // Number of experts (typically 8)
    top_k: u32,          // Number of experts to select (typically 2)
    normalize: u32,      // Whether to renormalize weights (1 = yes)
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> probs: array<f32>;           // [numTokens, numExperts]
@group(0) @binding(2) var<storage, read_write> out_indices: array<u32>; // [numTokens, topK]
@group(0) @binding(3) var<storage, read_write> out_weights: array<f32>; // [numTokens, topK]

// Workgroup shared memory for sorting
// Supports up to 256 experts (covers DeepSeek-V2's 160, Snowflake Arctic's 128, etc.)
var<workgroup> shared_probs: array<f32, 256>;
var<workgroup> shared_indices: array<u32, 256>;

// Main kernel: one workgroup per token
// Workgroup size 256 to support loading up to 256 experts in parallel
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let token_idx = wg_id.x;
    let thread_idx = local_id.x;
    let num_experts = u.num_experts;
    let top_k = u.top_k;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * num_experts;

    // Load probabilities into shared memory (first numExperts threads)
    if (thread_idx < num_experts) {
        shared_probs[thread_idx] = probs[base_offset + thread_idx];
        shared_indices[thread_idx] = thread_idx;
    }
    workgroupBarrier();

    // Simple selection sort for top-k (efficient for small k and n)
    // Only thread 0 does the sorting to avoid race conditions
    if (thread_idx == 0u) {
        // Find top-k by partial selection sort
        for (var k: u32 = 0u; k < top_k; k = k + 1u) {
            var max_idx = k;
            var max_val = shared_probs[k];

            // Find maximum in remaining elements
            for (var i: u32 = k + 1u; i < num_experts; i = i + 1u) {
                if (shared_probs[i] > max_val) {
                    max_val = shared_probs[i];
                    max_idx = i;
                }
            }

            // Swap if needed
            if (max_idx != k) {
                let tmp_prob = shared_probs[k];
                let tmp_idx = shared_indices[k];
                shared_probs[k] = shared_probs[max_idx];
                shared_indices[k] = shared_indices[max_idx];
                shared_probs[max_idx] = tmp_prob;
                shared_indices[max_idx] = tmp_idx;
            }
        }

        // Compute weight sum for normalization
        var weight_sum: f32 = 0.0;
        for (var k: u32 = 0u; k < top_k; k = k + 1u) {
            weight_sum = weight_sum + shared_probs[k];
        }

        // Write output indices and weights
        let out_base = token_idx * top_k;
        let inv_sum = select(1.0, 1.0 / weight_sum, u.normalize == 1u && weight_sum > 0.0);

        for (var k: u32 = 0u; k < top_k; k = k + 1u) {
            out_indices[out_base + k] = shared_indices[k];
            out_weights[out_base + k] = shared_probs[k] * inv_sum;
        }
    }
}

// Optimized version for topK=2, numExperts<=8
// Each thread handles one token
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn topk_2_small(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let num_experts = u.num_experts;
    let base_offset = token_idx * num_experts;

    // Find top 2 in a single pass
    var top1_idx: u32 = 0u;
    var top1_val: f32 = probs[base_offset];
    var top2_idx: u32 = 1u;
    var top2_val: f32 = probs[base_offset + 1u];

    // Ensure top1 >= top2
    if (top2_val > top1_val) {
        let tmp_idx = top1_idx;
        let tmp_val = top1_val;
        top1_idx = top2_idx;
        top1_val = top2_val;
        top2_idx = tmp_idx;
        top2_val = tmp_val;
    }

    // Scan remaining experts
    for (var i: u32 = 2u; i < num_experts; i = i + 1u) {
        let val = probs[base_offset + i];
        if (val > top1_val) {
            top2_idx = top1_idx;
            top2_val = top1_val;
            top1_idx = i;
            top1_val = val;
        } else if (val > top2_val) {
            top2_idx = i;
            top2_val = val;
        }
    }

    // Renormalize weights
    let weight_sum = top1_val + top2_val;
    let inv_sum = select(1.0, 1.0 / weight_sum, u.normalize == 1u && weight_sum > 0.0);

    // Write output
    let out_base = token_idx * 2u;
    out_indices[out_base] = top1_idx;
    out_indices[out_base + 1u] = top2_idx;
    out_weights[out_base] = top1_val * inv_sum;
    out_weights[out_base + 1u] = top2_val * inv_sum;
}

// Fused softmax + top-k for efficiency
// Avoids separate softmax kernel call
// Workgroup size 256 to support up to 256 experts
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn softmax_topk(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let token_idx = wg_id.x;
    let thread_idx = local_id.x;
    let num_experts = u.num_experts;
    let top_k = u.top_k;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * num_experts;

    // Load logits and find max (for numerical stability)
    if (thread_idx < num_experts) {
        shared_probs[thread_idx] = probs[base_offset + thread_idx];
        shared_indices[thread_idx] = thread_idx;
    }
    workgroupBarrier();

    // Thread 0 does softmax + top-k
    if (thread_idx == 0u) {
        // Find max
        var max_val: f32 = shared_probs[0];
        for (var i: u32 = 1u; i < num_experts; i = i + 1u) {
            max_val = max(max_val, shared_probs[i]);
        }

        // Compute exp and sum
        var exp_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < num_experts; i = i + 1u) {
            let exp_val = exp(shared_probs[i] - max_val);
            shared_probs[i] = exp_val;
            exp_sum = exp_sum + exp_val;
        }

        // Normalize to get probabilities
        let inv_exp_sum = 1.0 / exp_sum;
        for (var i: u32 = 0u; i < num_experts; i = i + 1u) {
            shared_probs[i] = shared_probs[i] * inv_exp_sum;
        }

        // Partial selection sort for top-k
        for (var k: u32 = 0u; k < top_k; k = k + 1u) {
            var max_idx = k;
            var max_prob = shared_probs[k];

            for (var i: u32 = k + 1u; i < num_experts; i = i + 1u) {
                if (shared_probs[i] > max_prob) {
                    max_prob = shared_probs[i];
                    max_idx = i;
                }
            }

            if (max_idx != k) {
                let tmp_prob = shared_probs[k];
                let tmp_idx = shared_indices[k];
                shared_probs[k] = shared_probs[max_idx];
                shared_indices[k] = shared_indices[max_idx];
                shared_probs[max_idx] = tmp_prob;
                shared_indices[max_idx] = tmp_idx;
            }
        }

        // Renormalize top-k weights
        var weight_sum: f32 = 0.0;
        for (var k: u32 = 0u; k < top_k; k = k + 1u) {
            weight_sum = weight_sum + shared_probs[k];
        }

        let out_base = token_idx * top_k;
        let inv_sum = select(1.0, 1.0 / weight_sum, u.normalize == 1u && weight_sum > 0.0);

        for (var k: u32 = 0u; k < top_k; k = k + 1u) {
            out_indices[out_base + k] = shared_indices[k];
            out_weights[out_base + k] = shared_probs[k] * inv_sum;
        }
    }
}
