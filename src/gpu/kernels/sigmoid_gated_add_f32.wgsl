// Explicit F32 scalar-per-row sigmoid gate, clamp, and residual add.
// Retains the unfused gate arithmetic; never adds the outer model residual.
override WORKGROUP_SIZE: u32 = 256u;
override HIDDEN_SIZE: u32;

struct Uniforms {
    size: u32,
    dispatch_stride: u32,
    clamp_max: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> residual: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn sigmoid(x: f32) -> f32 {
    let clamped = clamp(x, -15.0, 15.0);
    let z = exp_refined(-abs(clamped));
    let numerator = select(z, 1.0, clamped >= 0.0);
    let denominator = 1.0 + z;
    let inverse = reciprocal_refined(denominator);
    let quotient = numerator * inverse;
    return fma(fma(-quotient, denominator, numerator), inverse, quotient);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.y * u.dispatch_stride + gid.x;
    if (index >= u.size) { return; }
    let scale = sigmoid(gate[index / HIDDEN_SIZE]);
    var gated = scale * input[index];
    if (u.clamp_max > 0.0) {
        gated = clamp(gated, -u.clamp_max, u.clamp_max);
    }
    output[index] = residual[index] + gated;
}

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
