// Combines per-slot shared-slot down outputs in fixed slot order:
// output = sum_k route_weight[k] * routed[k] + clamp(sigmoid(gate) * shared).
// Same arithmetic order as the final step of moe_shared_slot_down_f32.wgsl.
override WORKGROUP_SIZE: u32 = 256u;
override TOP_K: u32;
struct Uniforms {
    num_tokens: u32, hidden_size: u32, intermediate_size: u32, shared_size: u32,
    num_experts: u32, slot_stride: u32, clamp_max: f32, _pad: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> slots: array<f32>;
@group(0) @binding(2) var<storage, read> route_weights: array<f32>;
@group(0) @binding(3) var<storage, read> scalar_gate: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
fn clamp_product(value: f32) -> f32 {
    if (u.clamp_max > 0.0) { return clamp(value, -u.clamp_max, u.clamp_max); }
    return value;
}
fn sigmoid(x: f32) -> f32 {
    let clamped = clamp(x, -15.0, 15.0);
    let z = exp_refined(-abs(clamped));
    let numerator = select(z, 1.0, clamped >= 0.0);
    let denominator = 1.0 + z;
    let inverse = reciprocal_refined(denominator);
    let quotient = numerator * inverse;
    return fma(fma(-quotient, denominator, numerator), inverse, quotient);
}

// Compensate the rounded numerator and quotient; retain the established gate clamp.
fn silu(x: f32) -> f32 {
    let z = exp_refined(-abs(clamp(x, -15.0, 15.0)));
 let numerator = select(x * z, x, x >= 0.0);
 let error = select(fma(x, z, -numerator), 0.0, x >= 0.0);
 let den = 1.0 + z;
 let inverse = reciprocal_refined(den);
 let q = numerator * inverse;
 return fma(fma(-q, den, numerator) + error, inverse, q);
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


@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>) {
    let index = gid.x + gid.y * grid.x * WORKGROUP_SIZE;
    if (index >= u.num_tokens * u.hidden_size) { return; }
    let token = index / u.hidden_size;
    let row = index % u.hidden_size;
    let base = token * (TOP_K + 1u) * u.hidden_size + row;
    var routed = 0.0;
    for (var slot = 0u; slot < TOP_K; slot++) {
        routed += route_weights[token * TOP_K + slot] * slots[base + slot * u.hidden_size];
    }
    let shared_value = clamp_product(sigmoid(scalar_gate[token]) * slots[base + TOP_K * u.hidden_size]);
    output[index] = routed + shared_value;
}
