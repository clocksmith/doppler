override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> grad_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> shared_sum_sq: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_sum_gx: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let token_idx = wid.x;
    if (token_idx >= u.num_tokens) {
        return;
    }

    let hidden_size = u.hidden_size;
    let base = token_idx * hidden_size;
    let local_id = lid.x;

    var sum_sq: f32 = 0.0;
    var sum_gx: f32 = 0.0;

    for (var i: u32 = local_id; i < hidden_size; i = i + WORKGROUP_SIZE) {
        let x = input[base + i];
        let g = grad_output[base + i] * weight[i];
        sum_sq = sum_sq + x * x;
        sum_gx = sum_gx + g * x;
    }

    shared_sum_sq[local_id] = sum_sq;
    shared_sum_gx[local_id] = sum_gx;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (local_id < stride) {
            shared_sum_sq[local_id] = shared_sum_sq[local_id] + shared_sum_sq[local_id + stride];
            shared_sum_gx[local_id] = shared_sum_gx[local_id] + shared_sum_gx[local_id + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let sum_sq_total = shared_sum_sq[0];
    let sum_gx_total = shared_sum_gx[0];
    let inv_rms = 1.0 / sqrt(sum_sq_total / f32(hidden_size) + u.eps);
    let inv_rms3 = inv_rms * inv_rms * inv_rms;
    let coeff = (sum_gx_total / f32(hidden_size)) * inv_rms3;

    for (var i: u32 = local_id; i < hidden_size; i = i + WORKGROUP_SIZE) {
        let x = input[base + i];
        let g = grad_output[base + i] * weight[i];
        output[base + i] = g * inv_rms - x * coeff;
    }
}
