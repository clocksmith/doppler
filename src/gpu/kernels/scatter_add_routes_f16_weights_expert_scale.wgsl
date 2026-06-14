// scatter_add_routes_f16_weights_expert_scale.wgsl

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct RouteUniforms {
    num_tokens: u32,
    hidden_size: u32,
    top_k: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: RouteUniforms;
@group(0) @binding(1) var<storage, read> route_outputs: array<f16>;
@group(0) @binding(2) var<storage, read> routing_indices: array<u32>;
@group(0) @binding(3) var<storage, read> routing_weights: array<f16>;
@group(0) @binding(4) var<storage, read> expert_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn scatter_add_routes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_elements = u.num_tokens * u.hidden_size;
    let base_tid = gid.x * 4u;

    for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {
        let tid = base_tid + lane;
        if (tid < total_elements) {
            let token_idx = tid / u.hidden_size;
            let dim_idx = tid % u.hidden_size;
            let route_base = token_idx * u.top_k;

            var sum: f16 = f16(0.0);
            for (var k: u32 = 0u; k < u.top_k; k = k + 1u) {
                let route_idx = route_base + k;
                let expert_idx = routing_indices[route_idx];
                let weight = routing_weights[route_idx] * f16(expert_scales[expert_idx]);
                let route_output_idx = route_idx * u.hidden_size + dim_idx;
                sum = sum + weight * route_outputs[route_output_idx];
            }

            output[tid] = sum;
        }
    }
}
