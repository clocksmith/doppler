/**
 * Scatter-Add Kernel (dynamic layout)
 *
 * Alternative layout: expert outputs stored per-expert with token batching.
 */

override WORKGROUP_SIZE: u32 = 256u;

// Alternative layout: expert outputs stored per-expert with token batching
// Layout: expertOutputs[expertIdx][batchedTokenIdx][hiddenSize]
// This version handles dynamic token-to-expert mapping
struct DynamicUniforms {
    num_tokens: u32,         // Total number of tokens
    hidden_size: u32,        // Hidden dimension
    top_k: u32,              // Number of experts per token
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u_dyn: DynamicUniforms;
@group(0) @binding(1) var<storage, read> expert_outputs_flat: array<f32>;  // Flattened expert outputs
@group(0) @binding(2) var<storage, read> routing_indices: array<u32>;      // [num_tokens, top_k] expert indices
@group(0) @binding(3) var<storage, read> routing_weights: array<f32>;      // [num_tokens, top_k] weights
@group(0) @binding(4) var<storage, read> token_offsets: array<u32>;        // Per-expert token offsets
@group(0) @binding(5) var<storage, read_write> output_dyn: array<f32>;     // [num_tokens, hidden_size]

// Dynamic scatter with per-expert token offset lookup
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn scatter_add_dynamic(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let total_elements = u_dyn.num_tokens * u_dyn.hidden_size;

    if (tid >= total_elements) {
        return;
    }

    let token_idx = tid / u_dyn.hidden_size;
    let dim_idx = tid % u_dyn.hidden_size;
    let top_k = u_dyn.top_k;
    let hidden_size = u_dyn.hidden_size;

    var sum: f32 = 0.0;
    let routing_base = token_idx * top_k;

    for (var k: u32 = 0u; k < top_k; k = k + 1u) {
        let expert_idx = routing_indices[routing_base + k];
        let weight = routing_weights[routing_base + k];

        // Look up where this token's data is stored for this expert
        // tokenOffsets[token_idx * top_k + k] gives the offset into expertOutputsFlat
        let data_offset = token_offsets[routing_base + k];
        let expert_data_idx = data_offset * hidden_size + dim_idx;

        sum = sum + weight * expert_outputs_flat[expert_data_idx];
    }

    output_dyn[tid] = sum;
}
