// gptoss_router_topk.wgsl

/**
 * GPT-OSS router top-k kernel.
 *
 * Identical bind-group topology to existing topk kernels, but with:
 * - deterministic tie-break by lower expert index,
 * - explicit f16 logits + f16 output weights,
 * - larger default workgroup for high-expert-count routing.
 */

enable f16;

override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_tokens: u32,
    num_experts: u32,
    top_k: u32,
    normalize: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> probs: array<f16>;
@group(0) @binding(2) var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_weights: array<f16>;

var<workgroup> shared_probs: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_indices: array<u32, MAX_WORKGROUP_SIZE>;

fn is_strictly_better(candidate_prob: f32, candidate_idx: u32, best_prob: f32, best_idx: u32) -> bool {
    if (candidate_prob > best_prob) {
        return true;
    }
    if (candidate_prob < best_prob) {
        return false;
    }
    return candidate_idx < best_idx;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn softmax_topk(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let token_idx = wg_id.x;
    let thread_idx = local_id.x;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let num_experts = u.num_experts;
    let top_k = u.top_k;
    let base_offset = token_idx * num_experts;

    if (thread_idx < num_experts) {
        shared_probs[thread_idx] = f32(probs[base_offset + thread_idx]);
        shared_indices[thread_idx] = thread_idx;
    }
    workgroupBarrier();

    if (thread_idx == 0u) {
        var max_val: f32 = shared_probs[0];
        for (var i: u32 = 1u; i < num_experts; i = i + 1u) {
            max_val = max(max_val, shared_probs[i]);
        }

        var exp_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < num_experts; i = i + 1u) {
            let exp_val = exp(shared_probs[i] - max_val);
            shared_probs[i] = exp_val;
            exp_sum = exp_sum + exp_val;
        }

        let inv_exp_sum = select(0.0, 1.0 / exp_sum, exp_sum > 0.0);
        for (var i: u32 = 0u; i < num_experts; i = i + 1u) {
            shared_probs[i] = shared_probs[i] * inv_exp_sum;
        }

        for (var k: u32 = 0u; k < top_k; k = k + 1u) {
            var best_idx: u32 = k;
            var best_prob: f32 = shared_probs[k];
            var best_expert: u32 = shared_indices[k];

            for (var i: u32 = k + 1u; i < num_experts; i = i + 1u) {
                let candidate_prob = shared_probs[i];
                let candidate_expert = shared_indices[i];
                if (is_strictly_better(candidate_prob, candidate_expert, best_prob, best_expert)) {
                    best_idx = i;
                    best_prob = candidate_prob;
                    best_expert = candidate_expert;
                }
            }

            if (best_idx != k) {
                let tmp_prob = shared_probs[k];
                let tmp_idx = shared_indices[k];
                shared_probs[k] = shared_probs[best_idx];
                shared_indices[k] = shared_indices[best_idx];
                shared_probs[best_idx] = tmp_prob;
                shared_indices[best_idx] = tmp_idx;
            }
        }

        var weight_sum: f32 = 0.0;
        for (var k: u32 = 0u; k < top_k; k = k + 1u) {
            weight_sum = weight_sum + shared_probs[k];
        }

        let out_base = token_idx * top_k;
        let inv_sum = select(1.0, 1.0 / weight_sum, u.normalize == 1u && weight_sum > 0.0);
        for (var k: u32 = 0u; k < top_k; k = k + 1u) {
            out_indices[out_base + k] = shared_indices[k];
            out_weights[out_base + k] = f16(shared_probs[k] * inv_sum);
        }
    }
}
