// Grouped RMSNorm over qwen4_exp hyper-connection streams.
// A token holds STREAM_COUNT streams of STREAM_WIDTH values. Each stream is
// normalized by its own root mean square, then scaled by (1 + weight) using
// that stream's weight slice (Qwen4ExpTextRMSNorm with group_size = width).

override WORKGROUP_SIZE: u32 = 256u;
override STREAM_COUNT: u32;
override STREAM_WIDTH: u32;

struct Uniforms {
    group_count: u32,   // numTokens * STREAM_COUNT
    group_stride: u32,  // workgroups dispatched along x
    eps: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> partial: array<f32, WORKGROUP_SIZE>;

fn inverse_root_refined(x: f32) -> f32 {
  let estimate = inverseSqrt(x);
  let squared = estimate * estimate;
  let error = fma(estimate, estimate, -squared);
  let residual = fma(-x, squared, 1.0) - x * error;
  return fma(0.5 * estimate, residual, estimate);
}

fn divide_refined(x: f32, y: f32) -> f32 {
  let estimate = 1.0 / y;
  let inverse = fma(estimate, fma(-y, estimate, 1.0), estimate);
  let quotient = x * inverse;
  return fma(fma(-quotient, y, x), inverse, quotient);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let group = wid.y * u.group_stride + wid.x;
    // Uniform per workgroup: every invocation of a group takes the same branch.
    if (group >= u.group_count) {
        return;
    }
    let local = lid.x;
    let base = group * STREAM_WIDTH;
    let weight_base = (group % STREAM_COUNT) * STREAM_WIDTH;

    var sum = 0.0;
    for (var column = local; column < STREAM_WIDTH; column = column + WORKGROUP_SIZE) {
        let value = input[base + column];
        sum = fma(value, value, sum);
    }
    partial[local] = sum;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (local < stride) {
            partial[local] = partial[local] + partial[local + stride];
        }
        workgroupBarrier();
    }
    let inverse_rms = inverse_root_refined(divide_refined(partial[0], f32(STREAM_WIDTH)) + u.eps);

    for (var column = local; column < STREAM_WIDTH; column = column + WORKGROUP_SIZE) {
        let normalized = input[base + column] * inverse_rms;
        output[base + column] = normalized * (1.0 + weight[weight_base + column]);
    }
}
