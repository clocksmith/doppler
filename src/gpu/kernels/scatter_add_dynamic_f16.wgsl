/**
 * Scatter-Add Kernel (dynamic layout, f16 outputs)
 */

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct DynamicUniforms {
    num_tokens: u32,
    hidden_size: u32,
    top_k: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u_dyn: DynamicUniforms;
@group(0) @binding(1) var<storage, read> expert_outputs_flat: array<f16>;
@group(0) @binding(2) var<storage, read> routing_indices: array<u32>;
@group(0) @binding(3) var<storage, read> routing_weights: array<f32>;
@group(0) @binding(4) var<storage, read> token_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> output_dyn: array<f16>;

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
        let weight = routing_weights[routing_base + k];
        let data_offset = token_offsets[routing_base + k];
        let expert_data_idx = data_offset * hidden_size + dim_idx;
        sum = sum + weight * f32(expert_outputs_flat[expert_data_idx]);
    }

    output_dyn[tid] = f16(sum);
}
