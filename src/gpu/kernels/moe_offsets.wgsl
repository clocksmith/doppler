// moe_offsets.wgsl
//
// Build per-routing-slot token offsets on GPU:
// token_offsets[token_idx * top_k + k_idx] = expert_idx * max_tokens_per_expert + slot_idx

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_tokens: u32,
    num_experts: u32,
    top_k: u32,
    max_tokens_per_expert: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> token_counts: array<u32>;
@group(0) @binding(2) var<storage, read> token_map: array<u32>;
@group(0) @binding(3) var<storage, read_write> token_offsets: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn build_offsets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot_linear = gid.x;
    let total_slots = u.num_experts * u.max_tokens_per_expert;
    if (slot_linear >= total_slots) {
        return;
    }

    let expert_idx = slot_linear / u.max_tokens_per_expert;
    let slot_idx = slot_linear % u.max_tokens_per_expert;
    let count = token_counts[expert_idx];
    if (slot_idx >= count) {
        return;
    }

    let map_base = (expert_idx * u.max_tokens_per_expert + slot_idx) * 2u;
    let token_idx = token_map[map_base];
    let k_idx = token_map[map_base + 1u];
    if (token_idx >= u.num_tokens || k_idx >= u.top_k) {
        return;
    }

    let routing_idx = token_idx * u.top_k + k_idx;
    token_offsets[routing_idx] = expert_idx * u.max_tokens_per_expert + slot_idx;
}
