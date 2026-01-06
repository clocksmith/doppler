/**
 * MoE Gather Kernel (vec4) - Gather tokens by expert for batched execution
 *
 * Optimized gather with vec4 loads.
 */

// Tunable workgroup size
override WORKGROUP_SIZE_VEC4: u32 = 64u;

struct Uniforms {
    num_tokens: u32,            // Number of input tokens
    hidden_size: u32,           // Hidden dimension
    num_experts: u32,           // Number of experts
    top_k: u32,                 // Number of experts per token
    max_tokens_per_expert: u32, // Max tokens any expert can receive
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> hidden_states: array<f32>;      // [num_tokens, hidden_size]
@group(0) @binding(2) var<storage, read> expert_indices: array<u32>;     // [num_tokens, top_k]
@group(0) @binding(3) var<storage, read_write> gathered: array<f32>;     // [num_experts, max_tokens_per_expert, hidden_size]
@group(0) @binding(4) var<storage, read_write> token_counts: array<atomic<u32>>; // [num_experts]
@group(0) @binding(5) var<storage, read_write> token_map: array<u32>;    // [num_experts, max_tokens_per_expert, 2]

// Optimized version: Gather with vec4 loads
@compute @workgroup_size(WORKGROUP_SIZE_VEC4, 1, 1)
fn gather_tokens_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let hidden_size = u.hidden_size;
    let max_tokens_per_expert = u.max_tokens_per_expert;
    let num_experts = u.num_experts;
    let vec4_per_token = hidden_size / 4u;

    let vec4_per_expert = max_tokens_per_expert * vec4_per_token;
    let total_vec4s = num_experts * vec4_per_expert;

    if (tid >= total_vec4s) {
        return;
    }

    // Decode position
    let expert_idx = tid / vec4_per_expert;
    let within_expert = tid % vec4_per_expert;
    let slot_idx = within_expert / vec4_per_token;
    let vec4_idx = within_expert % vec4_per_token;

    // Check if slot is valid
    let actual_count = atomicLoad(&token_counts[expert_idx]);
    let dst_base = tid * 4u;

    if (slot_idx >= actual_count) {
        gathered[dst_base] = 0.0;
        gathered[dst_base + 1u] = 0.0;
        gathered[dst_base + 2u] = 0.0;
        gathered[dst_base + 3u] = 0.0;
        return;
    }

    // Look up original token
    let map_base = expert_idx * max_tokens_per_expert * 2u + slot_idx * 2u;
    let token_idx = token_map[map_base];

    // Gather 4 elements
    let src_base = token_idx * hidden_size + vec4_idx * 4u;
    gathered[dst_base] = hidden_states[src_base];
    gathered[dst_base + 1u] = hidden_states[src_base + 1u];
    gathered[dst_base + 2u] = hidden_states[src_base + 2u];
    gathered[dst_base + 3u] = hidden_states[src_base + 3u];
}
