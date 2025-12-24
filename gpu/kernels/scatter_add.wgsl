/**
 * Scatter-Add Kernel for MoE Output Combination
 *
 * Combines expert outputs with weighted scatter-add operation.
 * Each token receives contributions from multiple experts weighted by routing probabilities.
 *
 * For MoE: output[token] = sum over k of (weight[token,k] * expert_output[expert[token,k], token])
 */

struct ScatterAddUniforms {
    numTokens: u32,      // Number of tokens
    hiddenSize: u32,     // Hidden dimension
    topK: u32,           // Number of experts per token
    numExperts: u32,     // Total number of experts
}

@group(0) @binding(0) var<uniform> uniforms: ScatterAddUniforms;
@group(0) @binding(1) var<storage, read> expertOutputs: array<f32>;  // [numExperts, numTokens, hiddenSize]
@group(0) @binding(2) var<storage, read> indices: array<u32>;         // [numTokens, topK]
@group(0) @binding(3) var<storage, read> weights: array<f32>;         // [numTokens, topK]
@group(0) @binding(4) var<storage, read_write> output: array<f32>;    // [numTokens, hiddenSize]

// Main kernel: each thread handles one output element
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let totalElements = uniforms.numTokens * uniforms.hiddenSize;

    if (tid >= totalElements) {
        return;
    }

    let tokenIdx = tid / uniforms.hiddenSize;
    let dimIdx = tid % uniforms.hiddenSize;
    let topK = uniforms.topK;
    let hiddenSize = uniforms.hiddenSize;
    let numTokens = uniforms.numTokens;

    // Accumulate weighted expert outputs
    var sum: f32 = 0.0;
    let routingBase = tokenIdx * topK;

    for (var k: u32 = 0u; k < topK; k = k + 1u) {
        let expertIdx = indices[routingBase + k];
        let weight = weights[routingBase + k];

        // Expert output layout: [numExperts, numTokens, hiddenSize]
        let expertOffset = expertIdx * numTokens * hiddenSize + tokenIdx * hiddenSize + dimIdx;
        sum = sum + weight * expertOutputs[expertOffset];
    }

    output[tid] = sum;
}

// Vectorized version (4 elements per thread)
@compute @workgroup_size(64, 1, 1)
fn scatter_add_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let vec4Count = uniforms.numTokens * (uniforms.hiddenSize / 4u);

    if (tid >= vec4Count) {
        return;
    }

    let hiddenSize = uniforms.hiddenSize;
    let numTokens = uniforms.numTokens;
    let topK = uniforms.topK;
    let vec4PerToken = hiddenSize / 4u;

    let tokenIdx = tid / vec4PerToken;
    let vec4Idx = tid % vec4PerToken;
    let dimBase = vec4Idx * 4u;

    // Accumulate weighted expert outputs
    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;

    let routingBase = tokenIdx * topK;

    for (var k: u32 = 0u; k < topK; k = k + 1u) {
        let expertIdx = indices[routingBase + k];
        let weight = weights[routingBase + k];

        // Expert output layout: [numExperts, numTokens, hiddenSize]
        let expertBase = expertIdx * numTokens * hiddenSize + tokenIdx * hiddenSize + dimBase;

        sum0 = sum0 + weight * expertOutputs[expertBase];
        sum1 = sum1 + weight * expertOutputs[expertBase + 1u];
        sum2 = sum2 + weight * expertOutputs[expertBase + 2u];
        sum3 = sum3 + weight * expertOutputs[expertBase + 3u];
    }

    let outBase = tokenIdx * hiddenSize + dimBase;
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
@compute @workgroup_size(256, 1, 1)
fn scatter_add_dynamic(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let totalElements = uniforms_dyn.numTokens * uniforms_dyn.hiddenSize;

    if (tid >= totalElements) {
        return;
    }

    let tokenIdx = tid / uniforms_dyn.hiddenSize;
    let dimIdx = tid % uniforms_dyn.hiddenSize;
    let topK = uniforms_dyn.topK;
    let hiddenSize = uniforms_dyn.hiddenSize;

    var sum: f32 = 0.0;
    let routingBase = tokenIdx * topK;

    for (var k: u32 = 0u; k < topK; k = k + 1u) {
        let expertIdx = routingIndices[routingBase + k];
        let weight = routingWeights[routingBase + k];

        // Look up where this token's data is stored for this expert
        // tokenOffsets[tokenIdx * topK + k] gives the offset into expertOutputsFlat
        let dataOffset = tokenOffsets[routingBase + k];
        let expertDataIdx = dataOffset * hiddenSize + dimIdx;

        sum = sum + weight * expertOutputsFlat[expertDataIdx];
    }

    outputDyn[tid] = sum;
}

// In-place accumulation version (adds to existing output)
@compute @workgroup_size(256, 1, 1)
fn scatter_add_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let totalElements = uniforms.numTokens * uniforms.hiddenSize;

    if (tid >= totalElements) {
        return;
    }

    let tokenIdx = tid / uniforms.hiddenSize;
    let dimIdx = tid % uniforms.hiddenSize;
    let topK = uniforms.topK;
    let hiddenSize = uniforms.hiddenSize;
    let numTokens = uniforms.numTokens;

    var sum: f32 = 0.0;
    let routingBase = tokenIdx * topK;

    for (var k: u32 = 0u; k < topK; k = k + 1u) {
        let expertIdx = indices[routingBase + k];
        let weight = weights[routingBase + k];

        let expertOffset = expertIdx * numTokens * hiddenSize + tokenIdx * hiddenSize + dimIdx;
        sum = sum + weight * expertOutputs[expertOffset];
    }

    // Accumulate to existing value
    output[tid] = output[tid] + sum;
}
