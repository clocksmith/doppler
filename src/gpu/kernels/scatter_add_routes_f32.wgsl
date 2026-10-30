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

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn scatter_add_routes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= u.num_tokens * u.hidden_size) { return; }
    let token = index / u.hidden_size;
    let channel = index % u.hidden_size;
    var sum = 0.0f;
    for (var k = 0u; k < u.top_k; k++) {
        let route = token * u.top_k + k;
        sum = sum + routing_weights[route] * route_outputs[route * u.hidden_size + channel];
    }
    output[index] = sum;
}
