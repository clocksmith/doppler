override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    count: u32,
    gate_mode: u32,
    swiglu_limit: f32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> grad_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_gate: array<f32>;

fn stable_sigmoid(x: f32) -> f32 {
    if (x >= 0.0) {
        let z = exp(-x);
        return 1.0 / (1.0 + z);
    }
    let z = exp(x);
    return z / (1.0 + z);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= u.count) {
        return;
    }
    let probability_input = select(gate[index], clamp(gate[index], -15.0, 15.0), u.gate_mode == 1u);
    let probability = stable_sigmoid(probability_input);
    let gradient = grad_output[index];
    if (u.gate_mode == 0u) {
        grad_input[index] = gradient * probability;
        grad_gate[index] = gradient * input[index] * probability * (1.0 - probability);
        return;
    }
    let activated = gate[index] * probability;
    if (u.swiglu_limit > 0.0 && abs(activated * input[index]) > u.swiglu_limit) {
        grad_input[index] = 0.0;
        grad_gate[index] = 0.0;
        return;
    }
    grad_input[index] = gradient * activated;
    grad_gate[index] = gradient * input[index]
        * probability * (1.0 + gate[index] * (1.0 - probability));
}
