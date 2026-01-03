// scatter_add.wgsl

/**
 * Scatter-Add Kernel for MoE Output Combination
 *
 * Combines expert outputs with weighted scatter-add operation.
 * Each token receives contributions from multiple experts weighted by routing probabilities.
 *
 * For MoE: output[token] = sum over k of (weight[token,k] * expert_output[expert[token,k], token])
 */

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_tokens: u32,     // Number of tokens
    hidden_size: u32,    // Hidden dimension
    top_k: u32,          // Number of experts per token
    num_experts: u32,    // Total number of experts
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> expert_outputs: array<f32>;  // [numExperts, numTokens, hiddenSize]
@group(0) @binding(2) var<storage, read> indices: array<u32>;         // [numTokens, topK]
@group(0) @binding(3) var<storage, read> weights: array<f32>;         // [numTokens, topK]
@group(0) @binding(4) var<storage, read_write> output: array<f32>;    // [numTokens, hiddenSize]

// Main kernel: each thread handles one output element
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let total_elements = u.num_tokens * u.hidden_size;

    if (tid >= total_elements) {
        return;
    }

    let token_idx = tid / u.hidden_size;
    let dim_idx = tid % u.hidden_size;
    let top_k = u.top_k;
    let hidden_size = u.hidden_size;
    let num_tokens = u.num_tokens;

    // Accumulate weighted expert outputs
    var sum: f32 = 0.0;
    let routing_base = token_idx * top_k;

    for (var k: u32 = 0u; k < top_k; k = k + 1u) {
        let expert_idx = indices[routing_base + k];
        let weight = weights[routing_base + k];

        // Expert output layout: [numExperts, numTokens, hiddenSize]
        let expert_offset = expert_idx * num_tokens * hidden_size + token_idx * hidden_size + dim_idx;
        sum = sum + weight * expert_outputs[expert_offset];
    }

    output[tid] = sum;
}

// In-place accumulation version (adds to existing output)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn scatter_add_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let total_elements = u.num_tokens * u.hidden_size;

    if (tid >= total_elements) {
        return;
    }

    let token_idx = tid / u.hidden_size;
    let dim_idx = tid % u.hidden_size;
    let top_k = u.top_k;
    let hidden_size = u.hidden_size;
    let num_tokens = u.num_tokens;

    var sum: f32 = 0.0;
    let routing_base = token_idx * top_k;

    for (var k: u32 = 0u; k < top_k; k = k + 1u) {
        let expert_idx = indices[routing_base + k];
        let weight = weights[routing_base + k];

        let expert_offset = expert_idx * num_tokens * hidden_size + token_idx * hidden_size + dim_idx;
        sum = sum + weight * expert_outputs[expert_offset];
    }

    // Accumulate to existing value
    output[tid] = output[tid] + sum;
}
