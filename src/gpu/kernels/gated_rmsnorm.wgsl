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
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> shared_sum_sq: array<f32, MAX_WORKGROUP_SIZE>;

fn silu(x: f32) -> f32 {
    if (x >= 0.0) {
        let z = exp(-x);
        return x / (1.0 + z);
    }
    let z = exp(x);
    return x * z / (1.0 + z);
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
    let lane = lid.x;
    let base = row * u.width;
    var sum_sq: f32 = 0.0;
    for (var column = lane; column < u.width; column = column + WORKGROUP_SIZE) {
        let value = input[base + column];
        sum_sq = sum_sq + value * value;
    }
    shared_sum_sq[lane] = sum_sq;
    workgroupBarrier();
    var stride = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            shared_sum_sq[lane] = shared_sum_sq[lane] + shared_sum_sq[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inverse_rms = inverseSqrt(shared_sum_sq[0] / f32(u.width) + u.eps);
    for (var column = lane; column < u.width; column = column + WORKGROUP_SIZE) {
        let offset = base + column;
        output[offset] = input[offset] * inverse_rms * weight[column] * silu(gate[offset]);
    }
}
