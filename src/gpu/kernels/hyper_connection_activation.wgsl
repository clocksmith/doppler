// Elementwise activations of the qwen4_exp gated residual (hyper-connections).
// low_rank_silu: silu(x / STREAM_COUNT) after the low-rank down projection.
// mix_sigmoid:   sigmoid(x) after the up projection (per-stream mix weights).
// inject_gate:   2 * sigmoid(x / STREAM_COUNT) for the branch injection weights.
// replicate:     streams[t, s, w] = embeds[t, w] (Qwen4ExpTextModel entry,
//                hidden_states.repeat(1, 1, hc_count)).
// Division is correctly rounded so any stream count matches source semantics.

override WORKGROUP_SIZE: u32 = 256u;
override STREAM_COUNT: u32;
override STREAM_WIDTH: u32 = 1u;  // used by replicate only

struct Uniforms {
    size: u32,
    dispatch_stride: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn exp_refined(x: f32) -> f32 {
  if (abs(x) > 80.0) {
    return exp(x);
  }
  let n = round(x * 1.4426950408889634);
  var r = x;
  if (abs(x) >= 0.000000059604644775390625) {
    let bits = bitcast<u32>(abs(x));
    let mantissa = (bits & 0x007fffffu) | 0x00800000u;
    let shift = i32((bits >> 23u) & 255u) - 118;
    var input_fixed = 0u;
    if (shift >= 0) {
      input_fixed = mantissa << u32(shift);
    } else {
      let right = u32(-shift);
      input_fixed = (mantissa + (1u << (right - 1u))) >> right;
    }
    let power_fixed = u32(abs(n)) * 2977044472u;
    let residual = select(power_fixed - input_fixed, input_fixed - power_fixed, x >= 0.0);
    r = f32(bitcast<i32>(residual)) * 0.00000000023283064365386962890625;
  }
  var p = 0.0001984126984126984;
  p = fma(p, r, 0.001388888888888889);
  p = fma(p, r, 0.008333333333333333);
  p = fma(p, r, 0.041666666666666664);
  p = fma(p, r, 0.16666666666666666);
  p = fma(p, r, 0.5);
  p = fma(p, r, 1.0);
  p = fma(p, r, 1.0);
  return p * bitcast<f32>(u32(i32(n) + 127) << 23u);
}

fn reciprocal_refined(x: f32) -> f32 {
  let estimate = 1.0 / x;
  return fma(estimate, fma(-x, estimate, 1.0), estimate);
}

fn divide_refined(x: f32, y: f32) -> f32 {
  let inverse = reciprocal_refined(y);
  let quotient = x * inverse;
  return fma(fma(-quotient, y, x), inverse, quotient);
}

fn sigmoid_refined(x: f32) -> f32 {
  let z = exp_refined(-abs(x));
  let reciprocal = reciprocal_refined(1.0 + z);
  return select(z * reciprocal, reciprocal, x >= 0.0);
}

fn silu(x: f32) -> f32 {
  let z = exp_refined(-abs(x));
  let numerator = select(x * z, x, x >= 0.0);
  return numerator * reciprocal_refined(1.0 + z);
}

fn element_index(gid: vec3<u32>) -> u32 {
  return gid.y * u.dispatch_stride + gid.x;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn low_rank_silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = element_index(gid);
    if (index >= u.size) { return; }
    output[index] = silu(divide_refined(input[index], f32(STREAM_COUNT)));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn mix_sigmoid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = element_index(gid);
    if (index >= u.size) { return; }
    output[index] = sigmoid_refined(input[index]);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn inject_gate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = element_index(gid);
    if (index >= u.size) { return; }
    output[index] = 2.0 * sigmoid_refined(divide_refined(input[index], f32(STREAM_COUNT)));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn replicate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = element_index(gid);
    if (index >= u.size) { return; }
    let token = index / (STREAM_COUNT * STREAM_WIDTH);
    output[index] = input[token * STREAM_WIDTH + index % STREAM_WIDTH];
}
