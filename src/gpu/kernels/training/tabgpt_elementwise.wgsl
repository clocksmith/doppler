// Dense GPT activation and tied embedding operations. Token batches are row-major.
override COUNT: u32 = 1u;
override WIDTH: u32 = 1u;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> other: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> grad_weights: array<atomic<u32>>;

@compute @workgroup_size(256)
fn embed(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < COUNT) {
        output[id.x] = input[ids[id.x / WIDTH] * WIDTH + id.x % WIDTH];
    }
}

// Add lookup gradients to the same table that already holds the LM-head gradient.
// Repeated tokens in a batch can update the same element concurrently.
@compute @workgroup_size(256)
fn embed_grad(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= COUNT) { return; }
    let index = ids[id.x / WIDTH] * WIDTH + id.x % WIDTH;
    let delta = input[id.x];
    var bits = atomicLoad(&grad_weights[index]);
    loop {
        let result = atomicCompareExchangeWeak(&grad_weights[index], bits,
            bitcast<u32>(bitcast<f32>(bits) + delta));
        if (result.exchanged) { break; }
        bits = result.old_value;
    }
}

@compute @workgroup_size(256)
fn relu_squared(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < COUNT) {
        let positive = max(0.0, input[id.x]);
        output[id.x] = positive * positive;
    }
}

@compute @workgroup_size(256)
fn relu_squared_grad(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < COUNT) {
        output[id.x] = input[id.x] * (2.0 * max(0.0, other[id.x]));
    }
}

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < COUNT) { output[id.x] = input[id.x] + other[id.x]; }
}

@compute @workgroup_size(256)
fn accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < COUNT) { output[id.x] += input[id.x]; }
}
