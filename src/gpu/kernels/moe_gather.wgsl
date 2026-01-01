/**
 * MoE Gather Kernel - Gather tokens by expert for batched execution
 *
 * Groups tokens by their selected experts so that each expert can
 * process its assigned tokens in a single batched operation.
 *
 * NOTE: This kernel requires EXPLICIT bind group layout (not 'auto')
 * because count_and_map and gather_tokens use different subsets of bindings.
 *
 * Input:
 *   - hidden_states [num_tokens, hidden_size]
 *   - indices [num_tokens, top_k] - selected expert indices per token
 *
 * Output:
 *   - gathered [num_experts, max_tokens_per_expert, hidden_size]
 *   - token_counts [num_experts] - actual token count per expert
 *   - token_map [num_experts, max_tokens_per_expert] - original token index mapping
 */

// Tunable workgroup sizes
override WORKGROUP_SIZE_MAIN: u32 = 256u;
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
@group(0) @binding(5) var<storage, read_write> token_map: array<u32>;    // [num_experts, max_tokens_per_expert, 2] (token_idx, k_idx)

// Phase 1: Count tokens per expert and build token map
// Run with num_tokens * top_k threads
@compute @workgroup_size(WORKGROUP_SIZE_MAIN, 1, 1)
fn count_and_map(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let total_slots = u.num_tokens * u.top_k;

    if (tid >= total_slots) {
        return;
    }

    let token_idx = tid / u.top_k;
    let k_idx = tid % u.top_k;
    let expert_idx = expert_indices[tid];

    // Atomically increment token count for this expert and get slot
    let slot = atomicAdd(&token_counts[expert_idx], 1u);

    // Store mapping: which original token goes to this slot
    if (slot < u.max_tokens_per_expert) {
        let map_base = expert_idx * u.max_tokens_per_expert * 2u + slot * 2u;
        token_map[map_base] = token_idx;
        token_map[map_base + 1u] = k_idx;
    }
}

// Phase 2: Gather hidden states based on token map
// Run with num_experts * max_tokens_per_expert * (hidden_size / 4) threads
@compute @workgroup_size(WORKGROUP_SIZE_MAIN, 1, 1)
fn gather_tokens(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let hidden_size = u.hidden_size;
    let max_tokens_per_expert = u.max_tokens_per_expert;
    let num_experts = u.num_experts;

    let elements_per_expert = max_tokens_per_expert * hidden_size;
    let total_elements = num_experts * elements_per_expert;

    if (tid >= total_elements) {
        return;
    }

    // Decode position
    let expert_idx = tid / elements_per_expert;
    let within_expert = tid % elements_per_expert;
    let slot_idx = within_expert / hidden_size;
    let dim_idx = within_expert % hidden_size;

    // Check if this slot is valid (within actual token count)
    let actual_count = atomicLoad(&token_counts[expert_idx]);
    if (slot_idx >= actual_count) {
        // Zero out unused slots
        gathered[tid] = 0.0;
        return;
    }

    // Look up original token index from map
    let map_base = expert_idx * max_tokens_per_expert * 2u + slot_idx * 2u;
    let token_idx = token_map[map_base];

    // Gather from original hidden states
    let src_idx = token_idx * hidden_size + dim_idx;
    gathered[tid] = hidden_states[src_idx];
}

// Combined single-pass version for small models
// Each workgroup handles one expert
@compute @workgroup_size(WORKGROUP_SIZE_MAIN, 1, 1)
fn gather_single_pass(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let expert_idx = wg_id.x;
    let thread_idx = local_id.x;
    let hidden_size = u.hidden_size;
    let num_tokens = u.num_tokens;
    let top_k = u.top_k;
    let max_tokens_per_expert = u.max_tokens_per_expert;

    if (expert_idx >= u.num_experts) {
        return;
    }

    // Phase 1: Count tokens for this expert (thread 0 only)
    var token_count: u32 = 0u;
    if (thread_idx == 0u) {
        for (var t: u32 = 0u; t < num_tokens; t = t + 1u) {
            for (var k: u32 = 0u; k < top_k; k = k + 1u) {
                if (expert_indices[t * top_k + k] == expert_idx) {
                    if (token_count < max_tokens_per_expert) {
                        let map_base = expert_idx * max_tokens_per_expert * 2u + token_count * 2u;
                        token_map[map_base] = t;
                        token_map[map_base + 1u] = k;
                        token_count = token_count + 1u;
                    }
                }
            }
        }
        atomicStore(&token_counts[expert_idx], token_count);
    }

    workgroupBarrier();

    // Phase 2: Gather (all threads participate)
    let actual_count = atomicLoad(&token_counts[expert_idx]);
    let elements_per_slot = hidden_size;
    let total_work = actual_count * elements_per_slot;
    let work_per_thread = (total_work + 255u) / 256u;

    for (var i: u32 = 0u; i < work_per_thread; i = i + 1u) {
        let work_idx = thread_idx * work_per_thread + i;
        if (work_idx >= total_work) {
            break;
        }

        let slot_idx = work_idx / hidden_size;
        let dim_idx = work_idx % hidden_size;

        let map_base = expert_idx * max_tokens_per_expert * 2u + slot_idx * 2u;
        let token_idx = token_map[map_base];

        let src_idx = token_idx * hidden_size + dim_idx;
        let dst_idx = expert_idx * max_tokens_per_expert * hidden_size + slot_idx * hidden_size + dim_idx;

        gathered[dst_idx] = hidden_states[src_idx];
    }
}

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
