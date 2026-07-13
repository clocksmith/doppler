override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    rows: u32,
    width: u32,
    eps: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read> grad_output: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_gate: array<f32>;

var<workgroup> shared_sum_sq: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_sum_gx: array<f32, MAX_WORKGROUP_SIZE>;

fn sigmoid(x: f32) -> f32 {
    if (x >= 0.0) {
        let z = exp(-x);
        return 1.0 / (1.0 + z);
    }
    let z = exp(x);
    return z / (1.0 + z);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wid.x;
    if (row >= u.rows) {
        return;
    }
    let local_id = lid.x;
    let base = row * u.width;
    var sum_sq: f32 = 0.0;
    var sum_gx: f32 = 0.0;

    for (var column = local_id; column < u.width; column = column + WORKGROUP_SIZE) {
        let offset = base + column;
        let x = input[offset];
        let probability = sigmoid(gate[offset]);
        let gate_value = gate[offset] * probability;
        let grad_normalized = grad_output[offset] * weight[column] * gate_value;
        sum_sq = sum_sq + x * x;
        sum_gx = sum_gx + grad_normalized * x;
    }

    shared_sum_sq[local_id] = sum_sq;
    shared_sum_gx[local_id] = sum_gx;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local_id < stride) {
            shared_sum_sq[local_id] = shared_sum_sq[local_id] + shared_sum_sq[local_id + stride];
            shared_sum_gx[local_id] = shared_sum_gx[local_id] + shared_sum_gx[local_id + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let inverse_rms = inverseSqrt(shared_sum_sq[0] / f32(u.width) + u.eps);
    let correction = (shared_sum_gx[0] * inverse_rms * inverse_rms) / f32(u.width);
    for (var column = local_id; column < u.width; column = column + WORKGROUP_SIZE) {
        let offset = base + column;
        let x = input[offset];
        let gate_input = gate[offset];
        let probability = sigmoid(gate_input);
        let gate_value = gate_input * probability;
        let gate_derivative = probability * (1.0 + gate_input * (1.0 - probability));
        let grad = grad_output[offset];
        let grad_normalized = grad * weight[column] * gate_value;
        grad_input[offset] = inverse_rms * (grad_normalized - x * correction);
        grad_gate[offset] = grad * x * inverse_rms * weight[column] * gate_derivative;
    }
}
