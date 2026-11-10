override WORKGROUP_SIZE: u32 = 256u;

struct RouteUniforms {
    num_tokens: u32,
    hidden_size: u32,
    top_k: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: RouteUniforms;
@group(0) @binding(1) var<storage, read> route_outputs: array<f32>;
@group(0) @binding(3) var<storage, read> routing_weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

// Source MoE materializes weighted expert outputs before adding them. Exchange
// products with an adjacent invocation so contraction cannot cross that boundary.
var<workgroup> rounded_products: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn scatter_add_routes(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let index = gid.x;
    let size = u.num_tokens * u.hidden_size;
    let group_base = index - lid.x;
    let product_index = group_base + (lid.x + 1u) % WORKGROUP_SIZE;
    let token = product_index / u.hidden_size;
    let channel = product_index % u.hidden_size;
    var sum = 0.0f;
    for (var k = 0u; k < u.top_k; k++) {
        let route = token * u.top_k + k;
        var product = 0.0f;
        if (product_index < size) {
            product = routing_weights[route] * route_outputs[route * u.hidden_size + channel];
        }
        rounded_products[lid.x] = product;
        workgroupBarrier();
        sum = sum + rounded_products[(lid.x + WORKGROUP_SIZE - 1u) % WORKGROUP_SIZE];
        workgroupBarrier();
    }
    if (index < size) { output[index] = sum; }
}
