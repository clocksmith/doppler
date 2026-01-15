/**
 * MoE Gather Kernel - f16 inputs/outputs
 */

enable f16;

override WORKGROUP_SIZE_MAIN: u32 = 256u;

struct Uniforms {
    num_tokens: u32,
    hidden_size: u32,
    num_experts: u32,
    top_k: u32,
    max_tokens_per_expert: u32,
    threads_per_row: u32,  // For 2D dispatch: dispatchSizeX * WORKGROUP_SIZE_MAIN
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> hidden_states: array<f16>;
@group(0) @binding(2) var<storage, read> expert_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> gathered: array<f16>;
@group(0) @binding(4) var<storage, read_write> token_counts: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> token_map: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE_MAIN, 1, 1)
fn count_and_map(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let total_slots = u.num_tokens * u.top_k;

    if (tid >= total_slots) {
        return;
    }

    let expert_idx = expert_indices[tid];
    let slot = atomicAdd(&token_counts[expert_idx], 1u);

    if (slot < u.max_tokens_per_expert) {
        let map_base = expert_idx * u.max_tokens_per_expert * 2u + slot * 2u;
        let token_idx = tid / u.top_k;
        let k_idx = tid % u.top_k;
        token_map[map_base] = token_idx;
        token_map[map_base + 1u] = k_idx;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE_MAIN, 1, 1)
fn gather_tokens(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Support 2D dispatch for large token counts (>65535 workgroups)
    let tid = gid.x + gid.y * u.threads_per_row;
    let hidden_size = u.hidden_size;
    let max_tokens_per_expert = u.max_tokens_per_expert;
    let num_experts = u.num_experts;

    let elements_per_expert = max_tokens_per_expert * hidden_size;
    let total_elements = num_experts * elements_per_expert;

    if (tid >= total_elements) {
        return;
    }

    let expert_idx = tid / elements_per_expert;
    let within_expert = tid % elements_per_expert;
    let slot_idx = within_expert / hidden_size;
    let dim_idx = within_expert % hidden_size;

    let actual_count = atomicLoad(&token_counts[expert_idx]);
    if (slot_idx >= actual_count) {
        gathered[tid] = f16(0.0);
        return;
    }

    let map_base = expert_idx * max_tokens_per_expert * 2u + slot_idx * 2u;
    let token_idx = token_map[map_base];
    let src_idx = token_idx * hidden_size + dim_idx;
    gathered[tid] = hidden_states[src_idx];
}
