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

// Vectorized version (4 elements per thread)
@compute @workgroup_size(WORKGROUP_SIZE / 4u, 1, 1)
fn scatter_add_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let vec4Count = u.num_tokens * (u.hidden_size / 4u);

    if (tid >= vec4Count) {
        return;
    }

    let hidden_size = u.hidden_size;
    let num_tokens = u.num_tokens;
    let top_k = u.top_k;
    let vec4PerToken = hiddenSize / 4u;

    let token_idx = tid / vec4PerToken;
    let vec4Idx = tid % vec4PerToken;
    let dimBase = vec4Idx * 4u;

    // Accumulate weighted expert outputs
    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;

    let routing_base = token_idx * top_k;

    for (var k: u32 = 0u; k < top_k; k = k + 1u) {
        let expert_idx = indices[routing_base + k];
        let weight = weights[routing_base + k];

        // Expert output layout: [numExperts, numTokens, hiddenSize]
        let expertBase = expert_idx * num_tokens * hidden_size + token_idx * hidden_size + dimBase;

        sum0 = sum0 + weight * expertOutputs[expertBase];
        sum1 = sum1 + weight * expertOutputs[expertBase + 1u];
        sum2 = sum2 + weight * expertOutputs[expertBase + 2u];
        sum3 = sum3 + weight * expertOutputs[expertBase + 3u];
    }

    let outBase = token_idx * hidden_size + dimBase;
    output[outBase] = sum0;
    output[outBase + 1u] = sum1;
    output[outBase + 2u] = sum2;
    output[outBase + 3u] = sum3;
}

// Alternative layout: expert outputs stored per-expert with token batching
// Layout: expertOutputs[expertIdx][batchedTokenIdx][hiddenSize]
// This version handles dynamic token-to-expert mapping
struct ScatterAddDynamicUniforms {
    numTokens: u32,          // Total number of tokens
    hiddenSize: u32,         // Hidden dimension
    topK: u32,               // Number of experts per token
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms_dyn: ScatterAddDynamicUniforms;
@group(0) @binding(1) var<storage, read> expertOutputsFlat: array<f32>;  // Flattened expert outputs
@group(0) @binding(2) var<storage, read> routingIndices: array<u32>;      // [numTokens, topK] expert indices
@group(0) @binding(3) var<storage, read> routingWeights: array<f32>;      // [numTokens, topK] weights
@group(0) @binding(4) var<storage, read> tokenOffsets: array<u32>;        // Per-expert token offsets
@group(0) @binding(5) var<storage, read_write> outputDyn: array<f32>;     // [numTokens, hiddenSize]

// Dynamic scatter with per-expert token offset lookup
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn scatter_add_dynamic(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let total_elements = uniforms_dyn.numTokens * uniforms_dyn.hiddenSize;

    if (tid >= total_elements) {
        return;
    }

    let token_idx = tid / uniforms_dyn.hiddenSize;
    let dim_idx = tid % uniforms_dyn.hiddenSize;
    let top_k = uniforms_dyn.topK;
    let hidden_size = uniforms_dyn.hiddenSize;

    var sum: f32 = 0.0;
    let routing_base = token_idx * top_k;

    for (var k: u32 = 0u; k < top_k; k = k + 1u) {
        let expert_idx = routingIndices[routingBase + k];
        let weight = routingWeights[routingBase + k];

        // Look up where this token's data is stored for this expert
        // tokenOffsets[token_idx * top_k + k] gives the offset into expertOutputsFlat
        let dataOffset = tokenOffsets[routingBase + k];
        let expertDataIdx = dataOffset * hidden_size + dim_idx;

        sum = sum + weight * expertOutputsFlat[expertDataIdx];
    }

    outputDyn[tid] = sum;
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
