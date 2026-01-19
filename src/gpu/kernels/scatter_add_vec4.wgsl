// scatter_add_vec4.wgsl

/**
 * Scatter-Add Kernel (vec4)
 *
 * Vectorized version (4 elements per thread).
 */

// Workgroup size for vec4 path
override WORKGROUP_SIZE_VEC4: u32 = 64u;

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

// Vectorized version (4 elements per thread)
@compute @workgroup_size(WORKGROUP_SIZE_VEC4, 1, 1)
fn scatter_add_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let vec4_count = u.num_tokens * (u.hidden_size / 4u);

    if (tid >= vec4_count) {
        return;
    }

    let hidden_size = u.hidden_size;
    let num_tokens = u.num_tokens;
    let top_k = u.top_k;
    let vec4_per_token = hidden_size / 4u;

    let token_idx = tid / vec4_per_token;
    let vec4_idx = tid % vec4_per_token;
    let dim_base = vec4_idx * 4u;

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
        let expert_base = expert_idx * num_tokens * hidden_size + token_idx * hidden_size + dim_base;

        sum0 = sum0 + weight * expert_outputs[expert_base];
        sum1 = sum1 + weight * expert_outputs[expert_base + 1u];
        sum2 = sum2 + weight * expert_outputs[expert_base + 2u];
        sum3 = sum3 + weight * expert_outputs[expert_base + 3u];
    }

    let out_base = token_idx * hidden_size + dim_base;
    output[out_base] = sum0;
    output[out_base + 1u] = sum1;
    output[out_base + 2u] = sum2;
    output[out_base + 3u] = sum3;
}
