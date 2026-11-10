// AUTO-GENERATED from src/gpu/kernels/gated_delta_recurrent.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// state without dynamic private-array indexing; register allocation is backend-dependent.
// Retention strategy informed by Pooled 6b4a24f73 engine/wgsl/qwen35.js dnDeltaRegsWGSL.
// GGUF arithmetic remains Doppler's ordered two-sweep recurrence, not Pooled's expanded output.
/*
MIT License

Copyright (c) 2026 Nehanth Narendrula

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
override RECURRENT_STATE_COMPONENTS: u32 = 1u;

override WORKGROUP_SIZE: u32 = 128u;
var<workgroup> shared_qk_sq: array<vec2<f32>, WORKGROUP_SIZE>;
var<workgroup> shared_bg: vec2<f32>;
var<workgroup> shared_q: array<f32, 128u>;
var<workgroup> shared_k: array<f32, 128u>;

struct LinearAttentionParams {
  num_tokens: u32,
  conv_dim: u32,
  conv_kernel_size: u32,
  num_v_heads: u32,
  num_k_heads: u32,
  head_k_dim: u32,
  head_v_dim: u32,
  q_size: u32,
  k_size: u32,
  value_dim: u32,
  q_rep: u32,
  norm_mode: u32,
  rms_norm_eps: f32,
  qk_l2norm_eps: f32,
  packed_flags: u32,
  b_proj_offset_elements: u32,
}

@group(0) @binding(0) var<uniform> params: LinearAttentionParams;
@group(0) @binding(1) var<storage, read> conv_out: array<f32>;
@group(0) @binding(2) var<storage, read> z_proj: array<f32>;
@group(0) @binding(3) var<storage, read> a_proj: array<f32>;
@group(0) @binding(4) var<storage, read> b_proj: array<f32>;
@group(0) @binding(5) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(6) var<storage, read> a_log: array<f32>;
@group(0) @binding(7) var<storage, read> norm_weight: array<f32>;
@group(0) @binding(8) var<storage, read_write> recurrent_state: array<f32>;
@group(0) @binding(9) var<storage, read_write> output: array<f32>;

var<workgroup> shared_sq: array<f32, WORKGROUP_SIZE>;

fn softplus(x: f32) -> f32 {
  if (x > 20.0) {
    return x;
  }
  if (x < -20.0) {
    return exp_refined(x);
  }
  let exponential = exp_refined(x);
  // Through e^5: at e <= 1/64 the omitted e^6/6 is below one F32 ULP.
  if (exponential <= 0.015625) {
    var polynomial = fma(exponential, 0.2, -0.25);
    polynomial = fma(polynomial, exponential, 0.3333333333333333);
    polynomial = fma(polynomial, exponential, -0.5);
    return fma(exponential * exponential, polynomial, exponential);
  }
  if (x <= 0.0) {
    // log1p(e) = 2*atanh(e/(2+e)); never round the small e away in 1+e.
    // Here e <= 1, so s <= 1/3. The omitted term is below 1.1e-11.
    let s = exponential * reciprocal_refined(2.0 + exponential);
    let squared = s * s;
    var polynomial = 0.05263157894736842;
    polynomial = fma(polynomial, squared, 0.058823529411764705);
    polynomial = fma(polynomial, squared, 0.06666666666666667);
    polynomial = fma(polynomial, squared, 0.07692307692307693);
    polynomial = fma(polynomial, squared, 0.09090909090909091);
    polynomial = fma(polynomial, squared, 0.1111111111111111);
    polynomial = fma(polynomial, squared, 0.14285714285714285);
    polynomial = fma(polynomial, squared, 0.2);
    polynomial = fma(polynomial, squared, 0.3333333333333333);
    polynomial = fma(polynomial, squared, 1.0);
    return (2.0 * s) * polynomial;
  }
  return log_refined(1.0 + exponential);
}

// Refine exp with a short F32 polynomial. Q32 range reduction prevents
// compiler reassociation of the split ln(2) subtraction. Only the low word
// is needed: the remainder lies in [-ln(2)/2, ln(2)/2], fitting signed Q32.
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

fn inverse_root_refined(x: f32) -> f32 {
  let estimate = inverseSqrt(x);
  let squared = estimate * estimate;
  let error = fma(estimate, estimate, -squared);
  let residual = fma(-x, squared, 1.0) - x * error;
  return fma(0.5 * estimate, residual, estimate);
}

fn silu(x: f32) -> f32 {
  let z = exp_refined(-abs(x));
  let numerator = select(x * z, x, x >= 0.0);
  return numerator * reciprocal_refined(1.0 + z);
}

fn sigmoid_refined(x: f32) -> f32 {
  let z = exp_refined(-abs(x));
  let reciprocal = reciprocal_refined(1.0 + z);
  return select(z * reciprocal, reciprocal, x >= 0.0);
}

// Softplus supplies positive normal log arguments. Mantissa reduction and
// the atanh series keep the F32 result close to the independent logarithm.
fn log_refined(x: f32) -> f32 {
  if (x <= 0.0 || x > 3.4028234663852886e38) {
    return log(x);
  }
  let bits = bitcast<u32>(x);
  var exponent = i32((bits >> 23u) & 255u) - 127;
  var mantissa = bitcast<f32>((bits & 0x007fffffu) | 0x3f800000u);
  if (mantissa > 1.4142135623730951) {
    mantissa = mantissa * 0.5;
    exponent = exponent + 1;
  }
  let s = (mantissa - 1.0) * reciprocal_refined(mantissa + 1.0);
  let squared = s * s;
  var p = 0.09090909090909091;
  p = fma(p, squared, 0.1111111111111111);
  p = fma(p, squared, 0.14285714285714285);
  p = fma(p, squared, 0.2);
  p = fma(p, squared, 0.3333333333333333);
  p = fma(p, squared, 1.0);
  let reduced = 2.0 * s * p;
  // Reconstruct exponent * ln(2) + reduced in signed Q32. Integer carry
  // and ties-to-even rounding make the final F32 rounding independent of
  // compiler contraction across the split-ln(2) expression. The rounded
  // ln(2) constant contributes < 6e-9 absolute error for normal F32 inputs.
  // exponent == 0 stays in F32 to preserve values arbitrarily close to one.
  if (exponent == 0) { return reduced; }
  let n = u32(abs(exponent));
  let bottom = n * (2977044472u & 65535u);
  let top = n * (2977044472u >> 16u);
  var lo = bottom + (top << 16u);
  var hi = (top >> 16u) + u32(lo < bottom);
  if (exponent < 0) {
    lo = 0u - lo;
    hi = ~hi + u32(lo == 0u);
  }
  let remainder = i32(round(reduced * 4294967296.0));
  let old_lo = lo;
  lo = lo + bitcast<u32>(remainder);
  hi = hi + select(0xffffffffu, 0u, remainder >= 0) + u32(lo < old_lo);
  let negative = (hi & 0x80000000u) != 0u;
  if (negative) {
    lo = 0u - lo;
    hi = ~hi + u32(lo == 0u);
  }
  let leading = select(firstLeadingBit(lo), 32u + firstLeadingBit(hi), hi != 0u);
  let shift = leading - 23u;
  var mant = (lo >> shift) | (hi << (32u - shift));
  let discarded = lo & ((1u << shift) - 1u);
  let halfway = 1u << (shift - 1u);
  mant = mant + u32(discarded > halfway || (discarded == halfway && (mant & 1u) != 0u));
  let magnitude = f32(mant) * bitcast<f32>((shift + 95u) << 23u);
  return select(magnitude, -magnitude, negative);
}

// Keep each scale product rounded before it enters a fused state update.
// The loop prevents cross-expression reassociation by the shader compiler.
fn multiply_ordered(a: f32, b: f32) -> f32 {
  var product = 1.0;
  let factors = array<f32, 2>(a, b);
  for (var i = 0u; i < 2u; i = i + 1u) {
    product = fma(product, factors[i], 0.0);
  }
  return product;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wid.x;
  let vd = lid.x;
  if (head >= params.num_v_heads) {
    return;
  }

  let head_k_dim = params.head_k_dim;
  let head_v_dim = params.head_v_dim;
  let is_active = vd < head_v_dim;
  let head_scale = inverseSqrt(f32(head_k_dim));
  let recurrent_head_base = head * head_k_dim * head_v_dim;
  let src_head = head % params.num_k_heads;
  let q_base = src_head * head_k_dim;
  let k_base = params.q_size + src_head * head_k_dim;
  let v_base = params.q_size + params.k_size + head * head_v_dim;

  var r0 = vec4<f32>(0.0); if (is_active) { r0 = vec4<f32>(recurrent_state[recurrent_head_base + 0u * head_v_dim + vd], recurrent_state[recurrent_head_base + 1u * head_v_dim + vd], recurrent_state[recurrent_head_base + 2u * head_v_dim + vd], recurrent_state[recurrent_head_base + 3u * head_v_dim + vd]); }
  var r1 = vec4<f32>(0.0); if (is_active) { r1 = vec4<f32>(recurrent_state[recurrent_head_base + 4u * head_v_dim + vd], recurrent_state[recurrent_head_base + 5u * head_v_dim + vd], recurrent_state[recurrent_head_base + 6u * head_v_dim + vd], recurrent_state[recurrent_head_base + 7u * head_v_dim + vd]); }
  var r2 = vec4<f32>(0.0); if (is_active) { r2 = vec4<f32>(recurrent_state[recurrent_head_base + 8u * head_v_dim + vd], recurrent_state[recurrent_head_base + 9u * head_v_dim + vd], recurrent_state[recurrent_head_base + 10u * head_v_dim + vd], recurrent_state[recurrent_head_base + 11u * head_v_dim + vd]); }
  var r3 = vec4<f32>(0.0); if (is_active) { r3 = vec4<f32>(recurrent_state[recurrent_head_base + 12u * head_v_dim + vd], recurrent_state[recurrent_head_base + 13u * head_v_dim + vd], recurrent_state[recurrent_head_base + 14u * head_v_dim + vd], recurrent_state[recurrent_head_base + 15u * head_v_dim + vd]); }
  var r4 = vec4<f32>(0.0); if (is_active) { r4 = vec4<f32>(recurrent_state[recurrent_head_base + 16u * head_v_dim + vd], recurrent_state[recurrent_head_base + 17u * head_v_dim + vd], recurrent_state[recurrent_head_base + 18u * head_v_dim + vd], recurrent_state[recurrent_head_base + 19u * head_v_dim + vd]); }
  var r5 = vec4<f32>(0.0); if (is_active) { r5 = vec4<f32>(recurrent_state[recurrent_head_base + 20u * head_v_dim + vd], recurrent_state[recurrent_head_base + 21u * head_v_dim + vd], recurrent_state[recurrent_head_base + 22u * head_v_dim + vd], recurrent_state[recurrent_head_base + 23u * head_v_dim + vd]); }
  var r6 = vec4<f32>(0.0); if (is_active) { r6 = vec4<f32>(recurrent_state[recurrent_head_base + 24u * head_v_dim + vd], recurrent_state[recurrent_head_base + 25u * head_v_dim + vd], recurrent_state[recurrent_head_base + 26u * head_v_dim + vd], recurrent_state[recurrent_head_base + 27u * head_v_dim + vd]); }
  var r7 = vec4<f32>(0.0); if (is_active) { r7 = vec4<f32>(recurrent_state[recurrent_head_base + 28u * head_v_dim + vd], recurrent_state[recurrent_head_base + 29u * head_v_dim + vd], recurrent_state[recurrent_head_base + 30u * head_v_dim + vd], recurrent_state[recurrent_head_base + 31u * head_v_dim + vd]); }
  var r8 = vec4<f32>(0.0); if (is_active) { r8 = vec4<f32>(recurrent_state[recurrent_head_base + 32u * head_v_dim + vd], recurrent_state[recurrent_head_base + 33u * head_v_dim + vd], recurrent_state[recurrent_head_base + 34u * head_v_dim + vd], recurrent_state[recurrent_head_base + 35u * head_v_dim + vd]); }
  var r9 = vec4<f32>(0.0); if (is_active) { r9 = vec4<f32>(recurrent_state[recurrent_head_base + 36u * head_v_dim + vd], recurrent_state[recurrent_head_base + 37u * head_v_dim + vd], recurrent_state[recurrent_head_base + 38u * head_v_dim + vd], recurrent_state[recurrent_head_base + 39u * head_v_dim + vd]); }
  var r10 = vec4<f32>(0.0); if (is_active) { r10 = vec4<f32>(recurrent_state[recurrent_head_base + 40u * head_v_dim + vd], recurrent_state[recurrent_head_base + 41u * head_v_dim + vd], recurrent_state[recurrent_head_base + 42u * head_v_dim + vd], recurrent_state[recurrent_head_base + 43u * head_v_dim + vd]); }
  var r11 = vec4<f32>(0.0); if (is_active) { r11 = vec4<f32>(recurrent_state[recurrent_head_base + 44u * head_v_dim + vd], recurrent_state[recurrent_head_base + 45u * head_v_dim + vd], recurrent_state[recurrent_head_base + 46u * head_v_dim + vd], recurrent_state[recurrent_head_base + 47u * head_v_dim + vd]); }
  var r12 = vec4<f32>(0.0); if (is_active) { r12 = vec4<f32>(recurrent_state[recurrent_head_base + 48u * head_v_dim + vd], recurrent_state[recurrent_head_base + 49u * head_v_dim + vd], recurrent_state[recurrent_head_base + 50u * head_v_dim + vd], recurrent_state[recurrent_head_base + 51u * head_v_dim + vd]); }
  var r13 = vec4<f32>(0.0); if (is_active) { r13 = vec4<f32>(recurrent_state[recurrent_head_base + 52u * head_v_dim + vd], recurrent_state[recurrent_head_base + 53u * head_v_dim + vd], recurrent_state[recurrent_head_base + 54u * head_v_dim + vd], recurrent_state[recurrent_head_base + 55u * head_v_dim + vd]); }
  var r14 = vec4<f32>(0.0); if (is_active) { r14 = vec4<f32>(recurrent_state[recurrent_head_base + 56u * head_v_dim + vd], recurrent_state[recurrent_head_base + 57u * head_v_dim + vd], recurrent_state[recurrent_head_base + 58u * head_v_dim + vd], recurrent_state[recurrent_head_base + 59u * head_v_dim + vd]); }
  var r15 = vec4<f32>(0.0); if (is_active) { r15 = vec4<f32>(recurrent_state[recurrent_head_base + 60u * head_v_dim + vd], recurrent_state[recurrent_head_base + 61u * head_v_dim + vd], recurrent_state[recurrent_head_base + 62u * head_v_dim + vd], recurrent_state[recurrent_head_base + 63u * head_v_dim + vd]); }
  var r16 = vec4<f32>(0.0); if (is_active) { r16 = vec4<f32>(recurrent_state[recurrent_head_base + 64u * head_v_dim + vd], recurrent_state[recurrent_head_base + 65u * head_v_dim + vd], recurrent_state[recurrent_head_base + 66u * head_v_dim + vd], recurrent_state[recurrent_head_base + 67u * head_v_dim + vd]); }
  var r17 = vec4<f32>(0.0); if (is_active) { r17 = vec4<f32>(recurrent_state[recurrent_head_base + 68u * head_v_dim + vd], recurrent_state[recurrent_head_base + 69u * head_v_dim + vd], recurrent_state[recurrent_head_base + 70u * head_v_dim + vd], recurrent_state[recurrent_head_base + 71u * head_v_dim + vd]); }
  var r18 = vec4<f32>(0.0); if (is_active) { r18 = vec4<f32>(recurrent_state[recurrent_head_base + 72u * head_v_dim + vd], recurrent_state[recurrent_head_base + 73u * head_v_dim + vd], recurrent_state[recurrent_head_base + 74u * head_v_dim + vd], recurrent_state[recurrent_head_base + 75u * head_v_dim + vd]); }
  var r19 = vec4<f32>(0.0); if (is_active) { r19 = vec4<f32>(recurrent_state[recurrent_head_base + 76u * head_v_dim + vd], recurrent_state[recurrent_head_base + 77u * head_v_dim + vd], recurrent_state[recurrent_head_base + 78u * head_v_dim + vd], recurrent_state[recurrent_head_base + 79u * head_v_dim + vd]); }
  var r20 = vec4<f32>(0.0); if (is_active) { r20 = vec4<f32>(recurrent_state[recurrent_head_base + 80u * head_v_dim + vd], recurrent_state[recurrent_head_base + 81u * head_v_dim + vd], recurrent_state[recurrent_head_base + 82u * head_v_dim + vd], recurrent_state[recurrent_head_base + 83u * head_v_dim + vd]); }
  var r21 = vec4<f32>(0.0); if (is_active) { r21 = vec4<f32>(recurrent_state[recurrent_head_base + 84u * head_v_dim + vd], recurrent_state[recurrent_head_base + 85u * head_v_dim + vd], recurrent_state[recurrent_head_base + 86u * head_v_dim + vd], recurrent_state[recurrent_head_base + 87u * head_v_dim + vd]); }
  var r22 = vec4<f32>(0.0); if (is_active) { r22 = vec4<f32>(recurrent_state[recurrent_head_base + 88u * head_v_dim + vd], recurrent_state[recurrent_head_base + 89u * head_v_dim + vd], recurrent_state[recurrent_head_base + 90u * head_v_dim + vd], recurrent_state[recurrent_head_base + 91u * head_v_dim + vd]); }
  var r23 = vec4<f32>(0.0); if (is_active) { r23 = vec4<f32>(recurrent_state[recurrent_head_base + 92u * head_v_dim + vd], recurrent_state[recurrent_head_base + 93u * head_v_dim + vd], recurrent_state[recurrent_head_base + 94u * head_v_dim + vd], recurrent_state[recurrent_head_base + 95u * head_v_dim + vd]); }
  var r24 = vec4<f32>(0.0); if (is_active) { r24 = vec4<f32>(recurrent_state[recurrent_head_base + 96u * head_v_dim + vd], recurrent_state[recurrent_head_base + 97u * head_v_dim + vd], recurrent_state[recurrent_head_base + 98u * head_v_dim + vd], recurrent_state[recurrent_head_base + 99u * head_v_dim + vd]); }
  var r25 = vec4<f32>(0.0); if (is_active) { r25 = vec4<f32>(recurrent_state[recurrent_head_base + 100u * head_v_dim + vd], recurrent_state[recurrent_head_base + 101u * head_v_dim + vd], recurrent_state[recurrent_head_base + 102u * head_v_dim + vd], recurrent_state[recurrent_head_base + 103u * head_v_dim + vd]); }
  var r26 = vec4<f32>(0.0); if (is_active) { r26 = vec4<f32>(recurrent_state[recurrent_head_base + 104u * head_v_dim + vd], recurrent_state[recurrent_head_base + 105u * head_v_dim + vd], recurrent_state[recurrent_head_base + 106u * head_v_dim + vd], recurrent_state[recurrent_head_base + 107u * head_v_dim + vd]); }
  var r27 = vec4<f32>(0.0); if (is_active) { r27 = vec4<f32>(recurrent_state[recurrent_head_base + 108u * head_v_dim + vd], recurrent_state[recurrent_head_base + 109u * head_v_dim + vd], recurrent_state[recurrent_head_base + 110u * head_v_dim + vd], recurrent_state[recurrent_head_base + 111u * head_v_dim + vd]); }
  var r28 = vec4<f32>(0.0); if (is_active) { r28 = vec4<f32>(recurrent_state[recurrent_head_base + 112u * head_v_dim + vd], recurrent_state[recurrent_head_base + 113u * head_v_dim + vd], recurrent_state[recurrent_head_base + 114u * head_v_dim + vd], recurrent_state[recurrent_head_base + 115u * head_v_dim + vd]); }
  var r29 = vec4<f32>(0.0); if (is_active) { r29 = vec4<f32>(recurrent_state[recurrent_head_base + 116u * head_v_dim + vd], recurrent_state[recurrent_head_base + 117u * head_v_dim + vd], recurrent_state[recurrent_head_base + 118u * head_v_dim + vd], recurrent_state[recurrent_head_base + 119u * head_v_dim + vd]); }
  var r30 = vec4<f32>(0.0); if (is_active) { r30 = vec4<f32>(recurrent_state[recurrent_head_base + 120u * head_v_dim + vd], recurrent_state[recurrent_head_base + 121u * head_v_dim + vd], recurrent_state[recurrent_head_base + 122u * head_v_dim + vd], recurrent_state[recurrent_head_base + 123u * head_v_dim + vd]); }
  var r31 = vec4<f32>(0.0); if (is_active) { r31 = vec4<f32>(recurrent_state[recurrent_head_base + 124u * head_v_dim + vd], recurrent_state[recurrent_head_base + 125u * head_v_dim + vd], recurrent_state[recurrent_head_base + 126u * head_v_dim + vd], recurrent_state[recurrent_head_base + 127u * head_v_dim + vd]); }

  for (var token_idx: u32 = 0u; token_idx < params.num_tokens; token_idx = token_idx + 1u) {
    let conv_row_base = token_idx * params.conv_dim;
    let z_row_base = token_idx * params.value_dim + head * head_v_dim;
    let z_packed_base = token_idx * (params.conv_dim + params.value_dim) + params.conv_dim + head * head_v_dim;
    let ab_row_base = token_idx * params.num_v_heads + head;
    let out_row_base = token_idx * params.value_dim + head * head_v_dim;

    let q_val = conv_out[conv_row_base + q_base + vd];
    let k_val = conv_out[conv_row_base + k_base + vd];
    shared_qk_sq[vd] = vec2<f32>(q_val * q_val, k_val * k_val);
    workgroupBarrier();
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
      if (vd < stride) {
        shared_qk_sq[vd] = shared_qk_sq[vd] + shared_qk_sq[vd + stride];
      }
      workgroupBarrier();
    }
    let q_norm_scale = head_scale * inverse_root_refined(shared_qk_sq[0].x + params.qk_l2norm_eps);
    let k_norm_scale = inverse_root_refined(shared_qk_sq[0].y + params.qk_l2norm_eps);
    shared_q[vd] = fma(q_val, q_norm_scale, 0.0);
    shared_k[vd] = multiply_ordered(k_val, k_norm_scale);
    if (vd == 0u) {
      let b_index = select(ab_row_base, params.b_proj_offset_elements + ab_row_base, (params.packed_flags & 1u) != 0u);
      let beta_val = sigmoid_refined(f32(b_proj[b_index]));
      let g = a_log[head] * softplus(f32(a_proj[ab_row_base]) + dt_bias[head]);
      shared_bg = vec2<f32>(beta_val, exp_refined(g));
    }
    workgroupBarrier();
    let beta = shared_bg.x;
    let g_exp = shared_bg.y;

    var kv_mem = 0.0;
    var out_value = 0.0;
    if (is_active) {
      { let k_normed = shared_k[0u]; r0.x = r0.x * g_exp; kv_mem = fma(r0.x, k_normed, kv_mem); }
      { let k_normed = shared_k[1u]; r0.y = r0.y * g_exp; kv_mem = fma(r0.y, k_normed, kv_mem); }
      { let k_normed = shared_k[2u]; r0.z = r0.z * g_exp; kv_mem = fma(r0.z, k_normed, kv_mem); }
      { let k_normed = shared_k[3u]; r0.w = r0.w * g_exp; kv_mem = fma(r0.w, k_normed, kv_mem); }
      { let k_normed = shared_k[4u]; r1.x = r1.x * g_exp; kv_mem = fma(r1.x, k_normed, kv_mem); }
      { let k_normed = shared_k[5u]; r1.y = r1.y * g_exp; kv_mem = fma(r1.y, k_normed, kv_mem); }
      { let k_normed = shared_k[6u]; r1.z = r1.z * g_exp; kv_mem = fma(r1.z, k_normed, kv_mem); }
      { let k_normed = shared_k[7u]; r1.w = r1.w * g_exp; kv_mem = fma(r1.w, k_normed, kv_mem); }
      { let k_normed = shared_k[8u]; r2.x = r2.x * g_exp; kv_mem = fma(r2.x, k_normed, kv_mem); }
      { let k_normed = shared_k[9u]; r2.y = r2.y * g_exp; kv_mem = fma(r2.y, k_normed, kv_mem); }
      { let k_normed = shared_k[10u]; r2.z = r2.z * g_exp; kv_mem = fma(r2.z, k_normed, kv_mem); }
      { let k_normed = shared_k[11u]; r2.w = r2.w * g_exp; kv_mem = fma(r2.w, k_normed, kv_mem); }
      { let k_normed = shared_k[12u]; r3.x = r3.x * g_exp; kv_mem = fma(r3.x, k_normed, kv_mem); }
      { let k_normed = shared_k[13u]; r3.y = r3.y * g_exp; kv_mem = fma(r3.y, k_normed, kv_mem); }
      { let k_normed = shared_k[14u]; r3.z = r3.z * g_exp; kv_mem = fma(r3.z, k_normed, kv_mem); }
      { let k_normed = shared_k[15u]; r3.w = r3.w * g_exp; kv_mem = fma(r3.w, k_normed, kv_mem); }
      { let k_normed = shared_k[16u]; r4.x = r4.x * g_exp; kv_mem = fma(r4.x, k_normed, kv_mem); }
      { let k_normed = shared_k[17u]; r4.y = r4.y * g_exp; kv_mem = fma(r4.y, k_normed, kv_mem); }
      { let k_normed = shared_k[18u]; r4.z = r4.z * g_exp; kv_mem = fma(r4.z, k_normed, kv_mem); }
      { let k_normed = shared_k[19u]; r4.w = r4.w * g_exp; kv_mem = fma(r4.w, k_normed, kv_mem); }
      { let k_normed = shared_k[20u]; r5.x = r5.x * g_exp; kv_mem = fma(r5.x, k_normed, kv_mem); }
      { let k_normed = shared_k[21u]; r5.y = r5.y * g_exp; kv_mem = fma(r5.y, k_normed, kv_mem); }
      { let k_normed = shared_k[22u]; r5.z = r5.z * g_exp; kv_mem = fma(r5.z, k_normed, kv_mem); }
      { let k_normed = shared_k[23u]; r5.w = r5.w * g_exp; kv_mem = fma(r5.w, k_normed, kv_mem); }
      { let k_normed = shared_k[24u]; r6.x = r6.x * g_exp; kv_mem = fma(r6.x, k_normed, kv_mem); }
      { let k_normed = shared_k[25u]; r6.y = r6.y * g_exp; kv_mem = fma(r6.y, k_normed, kv_mem); }
      { let k_normed = shared_k[26u]; r6.z = r6.z * g_exp; kv_mem = fma(r6.z, k_normed, kv_mem); }
      { let k_normed = shared_k[27u]; r6.w = r6.w * g_exp; kv_mem = fma(r6.w, k_normed, kv_mem); }
      { let k_normed = shared_k[28u]; r7.x = r7.x * g_exp; kv_mem = fma(r7.x, k_normed, kv_mem); }
      { let k_normed = shared_k[29u]; r7.y = r7.y * g_exp; kv_mem = fma(r7.y, k_normed, kv_mem); }
      { let k_normed = shared_k[30u]; r7.z = r7.z * g_exp; kv_mem = fma(r7.z, k_normed, kv_mem); }
      { let k_normed = shared_k[31u]; r7.w = r7.w * g_exp; kv_mem = fma(r7.w, k_normed, kv_mem); }
      { let k_normed = shared_k[32u]; r8.x = r8.x * g_exp; kv_mem = fma(r8.x, k_normed, kv_mem); }
      { let k_normed = shared_k[33u]; r8.y = r8.y * g_exp; kv_mem = fma(r8.y, k_normed, kv_mem); }
      { let k_normed = shared_k[34u]; r8.z = r8.z * g_exp; kv_mem = fma(r8.z, k_normed, kv_mem); }
      { let k_normed = shared_k[35u]; r8.w = r8.w * g_exp; kv_mem = fma(r8.w, k_normed, kv_mem); }
      { let k_normed = shared_k[36u]; r9.x = r9.x * g_exp; kv_mem = fma(r9.x, k_normed, kv_mem); }
      { let k_normed = shared_k[37u]; r9.y = r9.y * g_exp; kv_mem = fma(r9.y, k_normed, kv_mem); }
      { let k_normed = shared_k[38u]; r9.z = r9.z * g_exp; kv_mem = fma(r9.z, k_normed, kv_mem); }
      { let k_normed = shared_k[39u]; r9.w = r9.w * g_exp; kv_mem = fma(r9.w, k_normed, kv_mem); }
      { let k_normed = shared_k[40u]; r10.x = r10.x * g_exp; kv_mem = fma(r10.x, k_normed, kv_mem); }
      { let k_normed = shared_k[41u]; r10.y = r10.y * g_exp; kv_mem = fma(r10.y, k_normed, kv_mem); }
      { let k_normed = shared_k[42u]; r10.z = r10.z * g_exp; kv_mem = fma(r10.z, k_normed, kv_mem); }
      { let k_normed = shared_k[43u]; r10.w = r10.w * g_exp; kv_mem = fma(r10.w, k_normed, kv_mem); }
      { let k_normed = shared_k[44u]; r11.x = r11.x * g_exp; kv_mem = fma(r11.x, k_normed, kv_mem); }
      { let k_normed = shared_k[45u]; r11.y = r11.y * g_exp; kv_mem = fma(r11.y, k_normed, kv_mem); }
      { let k_normed = shared_k[46u]; r11.z = r11.z * g_exp; kv_mem = fma(r11.z, k_normed, kv_mem); }
      { let k_normed = shared_k[47u]; r11.w = r11.w * g_exp; kv_mem = fma(r11.w, k_normed, kv_mem); }
      { let k_normed = shared_k[48u]; r12.x = r12.x * g_exp; kv_mem = fma(r12.x, k_normed, kv_mem); }
      { let k_normed = shared_k[49u]; r12.y = r12.y * g_exp; kv_mem = fma(r12.y, k_normed, kv_mem); }
      { let k_normed = shared_k[50u]; r12.z = r12.z * g_exp; kv_mem = fma(r12.z, k_normed, kv_mem); }
      { let k_normed = shared_k[51u]; r12.w = r12.w * g_exp; kv_mem = fma(r12.w, k_normed, kv_mem); }
      { let k_normed = shared_k[52u]; r13.x = r13.x * g_exp; kv_mem = fma(r13.x, k_normed, kv_mem); }
      { let k_normed = shared_k[53u]; r13.y = r13.y * g_exp; kv_mem = fma(r13.y, k_normed, kv_mem); }
      { let k_normed = shared_k[54u]; r13.z = r13.z * g_exp; kv_mem = fma(r13.z, k_normed, kv_mem); }
      { let k_normed = shared_k[55u]; r13.w = r13.w * g_exp; kv_mem = fma(r13.w, k_normed, kv_mem); }
      { let k_normed = shared_k[56u]; r14.x = r14.x * g_exp; kv_mem = fma(r14.x, k_normed, kv_mem); }
      { let k_normed = shared_k[57u]; r14.y = r14.y * g_exp; kv_mem = fma(r14.y, k_normed, kv_mem); }
      { let k_normed = shared_k[58u]; r14.z = r14.z * g_exp; kv_mem = fma(r14.z, k_normed, kv_mem); }
      { let k_normed = shared_k[59u]; r14.w = r14.w * g_exp; kv_mem = fma(r14.w, k_normed, kv_mem); }
      { let k_normed = shared_k[60u]; r15.x = r15.x * g_exp; kv_mem = fma(r15.x, k_normed, kv_mem); }
      { let k_normed = shared_k[61u]; r15.y = r15.y * g_exp; kv_mem = fma(r15.y, k_normed, kv_mem); }
      { let k_normed = shared_k[62u]; r15.z = r15.z * g_exp; kv_mem = fma(r15.z, k_normed, kv_mem); }
      { let k_normed = shared_k[63u]; r15.w = r15.w * g_exp; kv_mem = fma(r15.w, k_normed, kv_mem); }
      { let k_normed = shared_k[64u]; r16.x = r16.x * g_exp; kv_mem = fma(r16.x, k_normed, kv_mem); }
      { let k_normed = shared_k[65u]; r16.y = r16.y * g_exp; kv_mem = fma(r16.y, k_normed, kv_mem); }
      { let k_normed = shared_k[66u]; r16.z = r16.z * g_exp; kv_mem = fma(r16.z, k_normed, kv_mem); }
      { let k_normed = shared_k[67u]; r16.w = r16.w * g_exp; kv_mem = fma(r16.w, k_normed, kv_mem); }
      { let k_normed = shared_k[68u]; r17.x = r17.x * g_exp; kv_mem = fma(r17.x, k_normed, kv_mem); }
      { let k_normed = shared_k[69u]; r17.y = r17.y * g_exp; kv_mem = fma(r17.y, k_normed, kv_mem); }
      { let k_normed = shared_k[70u]; r17.z = r17.z * g_exp; kv_mem = fma(r17.z, k_normed, kv_mem); }
      { let k_normed = shared_k[71u]; r17.w = r17.w * g_exp; kv_mem = fma(r17.w, k_normed, kv_mem); }
      { let k_normed = shared_k[72u]; r18.x = r18.x * g_exp; kv_mem = fma(r18.x, k_normed, kv_mem); }
      { let k_normed = shared_k[73u]; r18.y = r18.y * g_exp; kv_mem = fma(r18.y, k_normed, kv_mem); }
      { let k_normed = shared_k[74u]; r18.z = r18.z * g_exp; kv_mem = fma(r18.z, k_normed, kv_mem); }
      { let k_normed = shared_k[75u]; r18.w = r18.w * g_exp; kv_mem = fma(r18.w, k_normed, kv_mem); }
      { let k_normed = shared_k[76u]; r19.x = r19.x * g_exp; kv_mem = fma(r19.x, k_normed, kv_mem); }
      { let k_normed = shared_k[77u]; r19.y = r19.y * g_exp; kv_mem = fma(r19.y, k_normed, kv_mem); }
      { let k_normed = shared_k[78u]; r19.z = r19.z * g_exp; kv_mem = fma(r19.z, k_normed, kv_mem); }
      { let k_normed = shared_k[79u]; r19.w = r19.w * g_exp; kv_mem = fma(r19.w, k_normed, kv_mem); }
      { let k_normed = shared_k[80u]; r20.x = r20.x * g_exp; kv_mem = fma(r20.x, k_normed, kv_mem); }
      { let k_normed = shared_k[81u]; r20.y = r20.y * g_exp; kv_mem = fma(r20.y, k_normed, kv_mem); }
      { let k_normed = shared_k[82u]; r20.z = r20.z * g_exp; kv_mem = fma(r20.z, k_normed, kv_mem); }
      { let k_normed = shared_k[83u]; r20.w = r20.w * g_exp; kv_mem = fma(r20.w, k_normed, kv_mem); }
      { let k_normed = shared_k[84u]; r21.x = r21.x * g_exp; kv_mem = fma(r21.x, k_normed, kv_mem); }
      { let k_normed = shared_k[85u]; r21.y = r21.y * g_exp; kv_mem = fma(r21.y, k_normed, kv_mem); }
      { let k_normed = shared_k[86u]; r21.z = r21.z * g_exp; kv_mem = fma(r21.z, k_normed, kv_mem); }
      { let k_normed = shared_k[87u]; r21.w = r21.w * g_exp; kv_mem = fma(r21.w, k_normed, kv_mem); }
      { let k_normed = shared_k[88u]; r22.x = r22.x * g_exp; kv_mem = fma(r22.x, k_normed, kv_mem); }
      { let k_normed = shared_k[89u]; r22.y = r22.y * g_exp; kv_mem = fma(r22.y, k_normed, kv_mem); }
      { let k_normed = shared_k[90u]; r22.z = r22.z * g_exp; kv_mem = fma(r22.z, k_normed, kv_mem); }
      { let k_normed = shared_k[91u]; r22.w = r22.w * g_exp; kv_mem = fma(r22.w, k_normed, kv_mem); }
      { let k_normed = shared_k[92u]; r23.x = r23.x * g_exp; kv_mem = fma(r23.x, k_normed, kv_mem); }
      { let k_normed = shared_k[93u]; r23.y = r23.y * g_exp; kv_mem = fma(r23.y, k_normed, kv_mem); }
      { let k_normed = shared_k[94u]; r23.z = r23.z * g_exp; kv_mem = fma(r23.z, k_normed, kv_mem); }
      { let k_normed = shared_k[95u]; r23.w = r23.w * g_exp; kv_mem = fma(r23.w, k_normed, kv_mem); }
      { let k_normed = shared_k[96u]; r24.x = r24.x * g_exp; kv_mem = fma(r24.x, k_normed, kv_mem); }
      { let k_normed = shared_k[97u]; r24.y = r24.y * g_exp; kv_mem = fma(r24.y, k_normed, kv_mem); }
      { let k_normed = shared_k[98u]; r24.z = r24.z * g_exp; kv_mem = fma(r24.z, k_normed, kv_mem); }
      { let k_normed = shared_k[99u]; r24.w = r24.w * g_exp; kv_mem = fma(r24.w, k_normed, kv_mem); }
      { let k_normed = shared_k[100u]; r25.x = r25.x * g_exp; kv_mem = fma(r25.x, k_normed, kv_mem); }
      { let k_normed = shared_k[101u]; r25.y = r25.y * g_exp; kv_mem = fma(r25.y, k_normed, kv_mem); }
      { let k_normed = shared_k[102u]; r25.z = r25.z * g_exp; kv_mem = fma(r25.z, k_normed, kv_mem); }
      { let k_normed = shared_k[103u]; r25.w = r25.w * g_exp; kv_mem = fma(r25.w, k_normed, kv_mem); }
      { let k_normed = shared_k[104u]; r26.x = r26.x * g_exp; kv_mem = fma(r26.x, k_normed, kv_mem); }
      { let k_normed = shared_k[105u]; r26.y = r26.y * g_exp; kv_mem = fma(r26.y, k_normed, kv_mem); }
      { let k_normed = shared_k[106u]; r26.z = r26.z * g_exp; kv_mem = fma(r26.z, k_normed, kv_mem); }
      { let k_normed = shared_k[107u]; r26.w = r26.w * g_exp; kv_mem = fma(r26.w, k_normed, kv_mem); }
      { let k_normed = shared_k[108u]; r27.x = r27.x * g_exp; kv_mem = fma(r27.x, k_normed, kv_mem); }
      { let k_normed = shared_k[109u]; r27.y = r27.y * g_exp; kv_mem = fma(r27.y, k_normed, kv_mem); }
      { let k_normed = shared_k[110u]; r27.z = r27.z * g_exp; kv_mem = fma(r27.z, k_normed, kv_mem); }
      { let k_normed = shared_k[111u]; r27.w = r27.w * g_exp; kv_mem = fma(r27.w, k_normed, kv_mem); }
      { let k_normed = shared_k[112u]; r28.x = r28.x * g_exp; kv_mem = fma(r28.x, k_normed, kv_mem); }
      { let k_normed = shared_k[113u]; r28.y = r28.y * g_exp; kv_mem = fma(r28.y, k_normed, kv_mem); }
      { let k_normed = shared_k[114u]; r28.z = r28.z * g_exp; kv_mem = fma(r28.z, k_normed, kv_mem); }
      { let k_normed = shared_k[115u]; r28.w = r28.w * g_exp; kv_mem = fma(r28.w, k_normed, kv_mem); }
      { let k_normed = shared_k[116u]; r29.x = r29.x * g_exp; kv_mem = fma(r29.x, k_normed, kv_mem); }
      { let k_normed = shared_k[117u]; r29.y = r29.y * g_exp; kv_mem = fma(r29.y, k_normed, kv_mem); }
      { let k_normed = shared_k[118u]; r29.z = r29.z * g_exp; kv_mem = fma(r29.z, k_normed, kv_mem); }
      { let k_normed = shared_k[119u]; r29.w = r29.w * g_exp; kv_mem = fma(r29.w, k_normed, kv_mem); }
      { let k_normed = shared_k[120u]; r30.x = r30.x * g_exp; kv_mem = fma(r30.x, k_normed, kv_mem); }
      { let k_normed = shared_k[121u]; r30.y = r30.y * g_exp; kv_mem = fma(r30.y, k_normed, kv_mem); }
      { let k_normed = shared_k[122u]; r30.z = r30.z * g_exp; kv_mem = fma(r30.z, k_normed, kv_mem); }
      { let k_normed = shared_k[123u]; r30.w = r30.w * g_exp; kv_mem = fma(r30.w, k_normed, kv_mem); }
      { let k_normed = shared_k[124u]; r31.x = r31.x * g_exp; kv_mem = fma(r31.x, k_normed, kv_mem); }
      { let k_normed = shared_k[125u]; r31.y = r31.y * g_exp; kv_mem = fma(r31.y, k_normed, kv_mem); }
      { let k_normed = shared_k[126u]; r31.z = r31.z * g_exp; kv_mem = fma(r31.z, k_normed, kv_mem); }
      { let k_normed = shared_k[127u]; r31.w = r31.w * g_exp; kv_mem = fma(r31.w, k_normed, kv_mem); }
      let delta = multiply_ordered(conv_out[conv_row_base + v_base + vd] - kv_mem, beta);
      { let k_normed = shared_k[0u]; let q_normed = shared_q[0u]; r0.x = fma(k_normed, delta, r0.x); out_value = fma(r0.x, q_normed, out_value); }
      { let k_normed = shared_k[1u]; let q_normed = shared_q[1u]; r0.y = fma(k_normed, delta, r0.y); out_value = fma(r0.y, q_normed, out_value); }
      { let k_normed = shared_k[2u]; let q_normed = shared_q[2u]; r0.z = fma(k_normed, delta, r0.z); out_value = fma(r0.z, q_normed, out_value); }
      { let k_normed = shared_k[3u]; let q_normed = shared_q[3u]; r0.w = fma(k_normed, delta, r0.w); out_value = fma(r0.w, q_normed, out_value); }
      { let k_normed = shared_k[4u]; let q_normed = shared_q[4u]; r1.x = fma(k_normed, delta, r1.x); out_value = fma(r1.x, q_normed, out_value); }
      { let k_normed = shared_k[5u]; let q_normed = shared_q[5u]; r1.y = fma(k_normed, delta, r1.y); out_value = fma(r1.y, q_normed, out_value); }
      { let k_normed = shared_k[6u]; let q_normed = shared_q[6u]; r1.z = fma(k_normed, delta, r1.z); out_value = fma(r1.z, q_normed, out_value); }
      { let k_normed = shared_k[7u]; let q_normed = shared_q[7u]; r1.w = fma(k_normed, delta, r1.w); out_value = fma(r1.w, q_normed, out_value); }
      { let k_normed = shared_k[8u]; let q_normed = shared_q[8u]; r2.x = fma(k_normed, delta, r2.x); out_value = fma(r2.x, q_normed, out_value); }
      { let k_normed = shared_k[9u]; let q_normed = shared_q[9u]; r2.y = fma(k_normed, delta, r2.y); out_value = fma(r2.y, q_normed, out_value); }
      { let k_normed = shared_k[10u]; let q_normed = shared_q[10u]; r2.z = fma(k_normed, delta, r2.z); out_value = fma(r2.z, q_normed, out_value); }
      { let k_normed = shared_k[11u]; let q_normed = shared_q[11u]; r2.w = fma(k_normed, delta, r2.w); out_value = fma(r2.w, q_normed, out_value); }
      { let k_normed = shared_k[12u]; let q_normed = shared_q[12u]; r3.x = fma(k_normed, delta, r3.x); out_value = fma(r3.x, q_normed, out_value); }
      { let k_normed = shared_k[13u]; let q_normed = shared_q[13u]; r3.y = fma(k_normed, delta, r3.y); out_value = fma(r3.y, q_normed, out_value); }
      { let k_normed = shared_k[14u]; let q_normed = shared_q[14u]; r3.z = fma(k_normed, delta, r3.z); out_value = fma(r3.z, q_normed, out_value); }
      { let k_normed = shared_k[15u]; let q_normed = shared_q[15u]; r3.w = fma(k_normed, delta, r3.w); out_value = fma(r3.w, q_normed, out_value); }
      { let k_normed = shared_k[16u]; let q_normed = shared_q[16u]; r4.x = fma(k_normed, delta, r4.x); out_value = fma(r4.x, q_normed, out_value); }
      { let k_normed = shared_k[17u]; let q_normed = shared_q[17u]; r4.y = fma(k_normed, delta, r4.y); out_value = fma(r4.y, q_normed, out_value); }
      { let k_normed = shared_k[18u]; let q_normed = shared_q[18u]; r4.z = fma(k_normed, delta, r4.z); out_value = fma(r4.z, q_normed, out_value); }
      { let k_normed = shared_k[19u]; let q_normed = shared_q[19u]; r4.w = fma(k_normed, delta, r4.w); out_value = fma(r4.w, q_normed, out_value); }
      { let k_normed = shared_k[20u]; let q_normed = shared_q[20u]; r5.x = fma(k_normed, delta, r5.x); out_value = fma(r5.x, q_normed, out_value); }
      { let k_normed = shared_k[21u]; let q_normed = shared_q[21u]; r5.y = fma(k_normed, delta, r5.y); out_value = fma(r5.y, q_normed, out_value); }
      { let k_normed = shared_k[22u]; let q_normed = shared_q[22u]; r5.z = fma(k_normed, delta, r5.z); out_value = fma(r5.z, q_normed, out_value); }
      { let k_normed = shared_k[23u]; let q_normed = shared_q[23u]; r5.w = fma(k_normed, delta, r5.w); out_value = fma(r5.w, q_normed, out_value); }
      { let k_normed = shared_k[24u]; let q_normed = shared_q[24u]; r6.x = fma(k_normed, delta, r6.x); out_value = fma(r6.x, q_normed, out_value); }
      { let k_normed = shared_k[25u]; let q_normed = shared_q[25u]; r6.y = fma(k_normed, delta, r6.y); out_value = fma(r6.y, q_normed, out_value); }
      { let k_normed = shared_k[26u]; let q_normed = shared_q[26u]; r6.z = fma(k_normed, delta, r6.z); out_value = fma(r6.z, q_normed, out_value); }
      { let k_normed = shared_k[27u]; let q_normed = shared_q[27u]; r6.w = fma(k_normed, delta, r6.w); out_value = fma(r6.w, q_normed, out_value); }
      { let k_normed = shared_k[28u]; let q_normed = shared_q[28u]; r7.x = fma(k_normed, delta, r7.x); out_value = fma(r7.x, q_normed, out_value); }
      { let k_normed = shared_k[29u]; let q_normed = shared_q[29u]; r7.y = fma(k_normed, delta, r7.y); out_value = fma(r7.y, q_normed, out_value); }
      { let k_normed = shared_k[30u]; let q_normed = shared_q[30u]; r7.z = fma(k_normed, delta, r7.z); out_value = fma(r7.z, q_normed, out_value); }
      { let k_normed = shared_k[31u]; let q_normed = shared_q[31u]; r7.w = fma(k_normed, delta, r7.w); out_value = fma(r7.w, q_normed, out_value); }
      { let k_normed = shared_k[32u]; let q_normed = shared_q[32u]; r8.x = fma(k_normed, delta, r8.x); out_value = fma(r8.x, q_normed, out_value); }
      { let k_normed = shared_k[33u]; let q_normed = shared_q[33u]; r8.y = fma(k_normed, delta, r8.y); out_value = fma(r8.y, q_normed, out_value); }
      { let k_normed = shared_k[34u]; let q_normed = shared_q[34u]; r8.z = fma(k_normed, delta, r8.z); out_value = fma(r8.z, q_normed, out_value); }
      { let k_normed = shared_k[35u]; let q_normed = shared_q[35u]; r8.w = fma(k_normed, delta, r8.w); out_value = fma(r8.w, q_normed, out_value); }
      { let k_normed = shared_k[36u]; let q_normed = shared_q[36u]; r9.x = fma(k_normed, delta, r9.x); out_value = fma(r9.x, q_normed, out_value); }
      { let k_normed = shared_k[37u]; let q_normed = shared_q[37u]; r9.y = fma(k_normed, delta, r9.y); out_value = fma(r9.y, q_normed, out_value); }
      { let k_normed = shared_k[38u]; let q_normed = shared_q[38u]; r9.z = fma(k_normed, delta, r9.z); out_value = fma(r9.z, q_normed, out_value); }
      { let k_normed = shared_k[39u]; let q_normed = shared_q[39u]; r9.w = fma(k_normed, delta, r9.w); out_value = fma(r9.w, q_normed, out_value); }
      { let k_normed = shared_k[40u]; let q_normed = shared_q[40u]; r10.x = fma(k_normed, delta, r10.x); out_value = fma(r10.x, q_normed, out_value); }
      { let k_normed = shared_k[41u]; let q_normed = shared_q[41u]; r10.y = fma(k_normed, delta, r10.y); out_value = fma(r10.y, q_normed, out_value); }
      { let k_normed = shared_k[42u]; let q_normed = shared_q[42u]; r10.z = fma(k_normed, delta, r10.z); out_value = fma(r10.z, q_normed, out_value); }
      { let k_normed = shared_k[43u]; let q_normed = shared_q[43u]; r10.w = fma(k_normed, delta, r10.w); out_value = fma(r10.w, q_normed, out_value); }
      { let k_normed = shared_k[44u]; let q_normed = shared_q[44u]; r11.x = fma(k_normed, delta, r11.x); out_value = fma(r11.x, q_normed, out_value); }
      { let k_normed = shared_k[45u]; let q_normed = shared_q[45u]; r11.y = fma(k_normed, delta, r11.y); out_value = fma(r11.y, q_normed, out_value); }
      { let k_normed = shared_k[46u]; let q_normed = shared_q[46u]; r11.z = fma(k_normed, delta, r11.z); out_value = fma(r11.z, q_normed, out_value); }
      { let k_normed = shared_k[47u]; let q_normed = shared_q[47u]; r11.w = fma(k_normed, delta, r11.w); out_value = fma(r11.w, q_normed, out_value); }
      { let k_normed = shared_k[48u]; let q_normed = shared_q[48u]; r12.x = fma(k_normed, delta, r12.x); out_value = fma(r12.x, q_normed, out_value); }
      { let k_normed = shared_k[49u]; let q_normed = shared_q[49u]; r12.y = fma(k_normed, delta, r12.y); out_value = fma(r12.y, q_normed, out_value); }
      { let k_normed = shared_k[50u]; let q_normed = shared_q[50u]; r12.z = fma(k_normed, delta, r12.z); out_value = fma(r12.z, q_normed, out_value); }
      { let k_normed = shared_k[51u]; let q_normed = shared_q[51u]; r12.w = fma(k_normed, delta, r12.w); out_value = fma(r12.w, q_normed, out_value); }
      { let k_normed = shared_k[52u]; let q_normed = shared_q[52u]; r13.x = fma(k_normed, delta, r13.x); out_value = fma(r13.x, q_normed, out_value); }
      { let k_normed = shared_k[53u]; let q_normed = shared_q[53u]; r13.y = fma(k_normed, delta, r13.y); out_value = fma(r13.y, q_normed, out_value); }
      { let k_normed = shared_k[54u]; let q_normed = shared_q[54u]; r13.z = fma(k_normed, delta, r13.z); out_value = fma(r13.z, q_normed, out_value); }
      { let k_normed = shared_k[55u]; let q_normed = shared_q[55u]; r13.w = fma(k_normed, delta, r13.w); out_value = fma(r13.w, q_normed, out_value); }
      { let k_normed = shared_k[56u]; let q_normed = shared_q[56u]; r14.x = fma(k_normed, delta, r14.x); out_value = fma(r14.x, q_normed, out_value); }
      { let k_normed = shared_k[57u]; let q_normed = shared_q[57u]; r14.y = fma(k_normed, delta, r14.y); out_value = fma(r14.y, q_normed, out_value); }
      { let k_normed = shared_k[58u]; let q_normed = shared_q[58u]; r14.z = fma(k_normed, delta, r14.z); out_value = fma(r14.z, q_normed, out_value); }
      { let k_normed = shared_k[59u]; let q_normed = shared_q[59u]; r14.w = fma(k_normed, delta, r14.w); out_value = fma(r14.w, q_normed, out_value); }
      { let k_normed = shared_k[60u]; let q_normed = shared_q[60u]; r15.x = fma(k_normed, delta, r15.x); out_value = fma(r15.x, q_normed, out_value); }
      { let k_normed = shared_k[61u]; let q_normed = shared_q[61u]; r15.y = fma(k_normed, delta, r15.y); out_value = fma(r15.y, q_normed, out_value); }
      { let k_normed = shared_k[62u]; let q_normed = shared_q[62u]; r15.z = fma(k_normed, delta, r15.z); out_value = fma(r15.z, q_normed, out_value); }
      { let k_normed = shared_k[63u]; let q_normed = shared_q[63u]; r15.w = fma(k_normed, delta, r15.w); out_value = fma(r15.w, q_normed, out_value); }
      { let k_normed = shared_k[64u]; let q_normed = shared_q[64u]; r16.x = fma(k_normed, delta, r16.x); out_value = fma(r16.x, q_normed, out_value); }
      { let k_normed = shared_k[65u]; let q_normed = shared_q[65u]; r16.y = fma(k_normed, delta, r16.y); out_value = fma(r16.y, q_normed, out_value); }
      { let k_normed = shared_k[66u]; let q_normed = shared_q[66u]; r16.z = fma(k_normed, delta, r16.z); out_value = fma(r16.z, q_normed, out_value); }
      { let k_normed = shared_k[67u]; let q_normed = shared_q[67u]; r16.w = fma(k_normed, delta, r16.w); out_value = fma(r16.w, q_normed, out_value); }
      { let k_normed = shared_k[68u]; let q_normed = shared_q[68u]; r17.x = fma(k_normed, delta, r17.x); out_value = fma(r17.x, q_normed, out_value); }
      { let k_normed = shared_k[69u]; let q_normed = shared_q[69u]; r17.y = fma(k_normed, delta, r17.y); out_value = fma(r17.y, q_normed, out_value); }
      { let k_normed = shared_k[70u]; let q_normed = shared_q[70u]; r17.z = fma(k_normed, delta, r17.z); out_value = fma(r17.z, q_normed, out_value); }
      { let k_normed = shared_k[71u]; let q_normed = shared_q[71u]; r17.w = fma(k_normed, delta, r17.w); out_value = fma(r17.w, q_normed, out_value); }
      { let k_normed = shared_k[72u]; let q_normed = shared_q[72u]; r18.x = fma(k_normed, delta, r18.x); out_value = fma(r18.x, q_normed, out_value); }
      { let k_normed = shared_k[73u]; let q_normed = shared_q[73u]; r18.y = fma(k_normed, delta, r18.y); out_value = fma(r18.y, q_normed, out_value); }
      { let k_normed = shared_k[74u]; let q_normed = shared_q[74u]; r18.z = fma(k_normed, delta, r18.z); out_value = fma(r18.z, q_normed, out_value); }
      { let k_normed = shared_k[75u]; let q_normed = shared_q[75u]; r18.w = fma(k_normed, delta, r18.w); out_value = fma(r18.w, q_normed, out_value); }
      { let k_normed = shared_k[76u]; let q_normed = shared_q[76u]; r19.x = fma(k_normed, delta, r19.x); out_value = fma(r19.x, q_normed, out_value); }
      { let k_normed = shared_k[77u]; let q_normed = shared_q[77u]; r19.y = fma(k_normed, delta, r19.y); out_value = fma(r19.y, q_normed, out_value); }
      { let k_normed = shared_k[78u]; let q_normed = shared_q[78u]; r19.z = fma(k_normed, delta, r19.z); out_value = fma(r19.z, q_normed, out_value); }
      { let k_normed = shared_k[79u]; let q_normed = shared_q[79u]; r19.w = fma(k_normed, delta, r19.w); out_value = fma(r19.w, q_normed, out_value); }
      { let k_normed = shared_k[80u]; let q_normed = shared_q[80u]; r20.x = fma(k_normed, delta, r20.x); out_value = fma(r20.x, q_normed, out_value); }
      { let k_normed = shared_k[81u]; let q_normed = shared_q[81u]; r20.y = fma(k_normed, delta, r20.y); out_value = fma(r20.y, q_normed, out_value); }
      { let k_normed = shared_k[82u]; let q_normed = shared_q[82u]; r20.z = fma(k_normed, delta, r20.z); out_value = fma(r20.z, q_normed, out_value); }
      { let k_normed = shared_k[83u]; let q_normed = shared_q[83u]; r20.w = fma(k_normed, delta, r20.w); out_value = fma(r20.w, q_normed, out_value); }
      { let k_normed = shared_k[84u]; let q_normed = shared_q[84u]; r21.x = fma(k_normed, delta, r21.x); out_value = fma(r21.x, q_normed, out_value); }
      { let k_normed = shared_k[85u]; let q_normed = shared_q[85u]; r21.y = fma(k_normed, delta, r21.y); out_value = fma(r21.y, q_normed, out_value); }
      { let k_normed = shared_k[86u]; let q_normed = shared_q[86u]; r21.z = fma(k_normed, delta, r21.z); out_value = fma(r21.z, q_normed, out_value); }
      { let k_normed = shared_k[87u]; let q_normed = shared_q[87u]; r21.w = fma(k_normed, delta, r21.w); out_value = fma(r21.w, q_normed, out_value); }
      { let k_normed = shared_k[88u]; let q_normed = shared_q[88u]; r22.x = fma(k_normed, delta, r22.x); out_value = fma(r22.x, q_normed, out_value); }
      { let k_normed = shared_k[89u]; let q_normed = shared_q[89u]; r22.y = fma(k_normed, delta, r22.y); out_value = fma(r22.y, q_normed, out_value); }
      { let k_normed = shared_k[90u]; let q_normed = shared_q[90u]; r22.z = fma(k_normed, delta, r22.z); out_value = fma(r22.z, q_normed, out_value); }
      { let k_normed = shared_k[91u]; let q_normed = shared_q[91u]; r22.w = fma(k_normed, delta, r22.w); out_value = fma(r22.w, q_normed, out_value); }
      { let k_normed = shared_k[92u]; let q_normed = shared_q[92u]; r23.x = fma(k_normed, delta, r23.x); out_value = fma(r23.x, q_normed, out_value); }
      { let k_normed = shared_k[93u]; let q_normed = shared_q[93u]; r23.y = fma(k_normed, delta, r23.y); out_value = fma(r23.y, q_normed, out_value); }
      { let k_normed = shared_k[94u]; let q_normed = shared_q[94u]; r23.z = fma(k_normed, delta, r23.z); out_value = fma(r23.z, q_normed, out_value); }
      { let k_normed = shared_k[95u]; let q_normed = shared_q[95u]; r23.w = fma(k_normed, delta, r23.w); out_value = fma(r23.w, q_normed, out_value); }
      { let k_normed = shared_k[96u]; let q_normed = shared_q[96u]; r24.x = fma(k_normed, delta, r24.x); out_value = fma(r24.x, q_normed, out_value); }
      { let k_normed = shared_k[97u]; let q_normed = shared_q[97u]; r24.y = fma(k_normed, delta, r24.y); out_value = fma(r24.y, q_normed, out_value); }
      { let k_normed = shared_k[98u]; let q_normed = shared_q[98u]; r24.z = fma(k_normed, delta, r24.z); out_value = fma(r24.z, q_normed, out_value); }
      { let k_normed = shared_k[99u]; let q_normed = shared_q[99u]; r24.w = fma(k_normed, delta, r24.w); out_value = fma(r24.w, q_normed, out_value); }
      { let k_normed = shared_k[100u]; let q_normed = shared_q[100u]; r25.x = fma(k_normed, delta, r25.x); out_value = fma(r25.x, q_normed, out_value); }
      { let k_normed = shared_k[101u]; let q_normed = shared_q[101u]; r25.y = fma(k_normed, delta, r25.y); out_value = fma(r25.y, q_normed, out_value); }
      { let k_normed = shared_k[102u]; let q_normed = shared_q[102u]; r25.z = fma(k_normed, delta, r25.z); out_value = fma(r25.z, q_normed, out_value); }
      { let k_normed = shared_k[103u]; let q_normed = shared_q[103u]; r25.w = fma(k_normed, delta, r25.w); out_value = fma(r25.w, q_normed, out_value); }
      { let k_normed = shared_k[104u]; let q_normed = shared_q[104u]; r26.x = fma(k_normed, delta, r26.x); out_value = fma(r26.x, q_normed, out_value); }
      { let k_normed = shared_k[105u]; let q_normed = shared_q[105u]; r26.y = fma(k_normed, delta, r26.y); out_value = fma(r26.y, q_normed, out_value); }
      { let k_normed = shared_k[106u]; let q_normed = shared_q[106u]; r26.z = fma(k_normed, delta, r26.z); out_value = fma(r26.z, q_normed, out_value); }
      { let k_normed = shared_k[107u]; let q_normed = shared_q[107u]; r26.w = fma(k_normed, delta, r26.w); out_value = fma(r26.w, q_normed, out_value); }
      { let k_normed = shared_k[108u]; let q_normed = shared_q[108u]; r27.x = fma(k_normed, delta, r27.x); out_value = fma(r27.x, q_normed, out_value); }
      { let k_normed = shared_k[109u]; let q_normed = shared_q[109u]; r27.y = fma(k_normed, delta, r27.y); out_value = fma(r27.y, q_normed, out_value); }
      { let k_normed = shared_k[110u]; let q_normed = shared_q[110u]; r27.z = fma(k_normed, delta, r27.z); out_value = fma(r27.z, q_normed, out_value); }
      { let k_normed = shared_k[111u]; let q_normed = shared_q[111u]; r27.w = fma(k_normed, delta, r27.w); out_value = fma(r27.w, q_normed, out_value); }
      { let k_normed = shared_k[112u]; let q_normed = shared_q[112u]; r28.x = fma(k_normed, delta, r28.x); out_value = fma(r28.x, q_normed, out_value); }
      { let k_normed = shared_k[113u]; let q_normed = shared_q[113u]; r28.y = fma(k_normed, delta, r28.y); out_value = fma(r28.y, q_normed, out_value); }
      { let k_normed = shared_k[114u]; let q_normed = shared_q[114u]; r28.z = fma(k_normed, delta, r28.z); out_value = fma(r28.z, q_normed, out_value); }
      { let k_normed = shared_k[115u]; let q_normed = shared_q[115u]; r28.w = fma(k_normed, delta, r28.w); out_value = fma(r28.w, q_normed, out_value); }
      { let k_normed = shared_k[116u]; let q_normed = shared_q[116u]; r29.x = fma(k_normed, delta, r29.x); out_value = fma(r29.x, q_normed, out_value); }
      { let k_normed = shared_k[117u]; let q_normed = shared_q[117u]; r29.y = fma(k_normed, delta, r29.y); out_value = fma(r29.y, q_normed, out_value); }
      { let k_normed = shared_k[118u]; let q_normed = shared_q[118u]; r29.z = fma(k_normed, delta, r29.z); out_value = fma(r29.z, q_normed, out_value); }
      { let k_normed = shared_k[119u]; let q_normed = shared_q[119u]; r29.w = fma(k_normed, delta, r29.w); out_value = fma(r29.w, q_normed, out_value); }
      { let k_normed = shared_k[120u]; let q_normed = shared_q[120u]; r30.x = fma(k_normed, delta, r30.x); out_value = fma(r30.x, q_normed, out_value); }
      { let k_normed = shared_k[121u]; let q_normed = shared_q[121u]; r30.y = fma(k_normed, delta, r30.y); out_value = fma(r30.y, q_normed, out_value); }
      { let k_normed = shared_k[122u]; let q_normed = shared_q[122u]; r30.z = fma(k_normed, delta, r30.z); out_value = fma(r30.z, q_normed, out_value); }
      { let k_normed = shared_k[123u]; let q_normed = shared_q[123u]; r30.w = fma(k_normed, delta, r30.w); out_value = fma(r30.w, q_normed, out_value); }
      { let k_normed = shared_k[124u]; let q_normed = shared_q[124u]; r31.x = fma(k_normed, delta, r31.x); out_value = fma(r31.x, q_normed, out_value); }
      { let k_normed = shared_k[125u]; let q_normed = shared_q[125u]; r31.y = fma(k_normed, delta, r31.y); out_value = fma(r31.y, q_normed, out_value); }
      { let k_normed = shared_k[126u]; let q_normed = shared_q[126u]; r31.z = fma(k_normed, delta, r31.z); out_value = fma(r31.z, q_normed, out_value); }
      { let k_normed = shared_k[127u]; let q_normed = shared_q[127u]; r31.w = fma(k_normed, delta, r31.w); out_value = fma(r31.w, q_normed, out_value); }
    }
    shared_sq[vd] = select(0.0, out_value * out_value, is_active);
    workgroupBarrier();
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
      if (vd < stride) {
        shared_sq[vd] = shared_sq[vd] + shared_sq[vd + stride];
      }
      workgroupBarrier();
    }
    let inv_rms = inverse_root_refined(shared_sq[0] / f32(head_v_dim) + params.rms_norm_eps);

    if (is_active) {
      let z_index = select(z_row_base + vd, z_packed_base + vd, (params.packed_flags & 2u) != 0u);
      let gate = silu(f32(z_proj[z_index]));
      let norm_index = select(vd, head * head_v_dim + vd, params.norm_mode == 1u);
      var value = out_value;
      let factors = array<f32, 3>(inv_rms, norm_weight[norm_index], gate);
      for (var i = 0u; i < 3u; i = i + 1u) {
        value = fma(value, factors[i], 0.0);
      }
      output[out_row_base + vd] = value;
    }
  }
  if (is_active) {
    recurrent_state[recurrent_head_base + 0u * head_v_dim + vd] = r0.x;
    recurrent_state[recurrent_head_base + 1u * head_v_dim + vd] = r0.y;
    recurrent_state[recurrent_head_base + 2u * head_v_dim + vd] = r0.z;
    recurrent_state[recurrent_head_base + 3u * head_v_dim + vd] = r0.w;
    recurrent_state[recurrent_head_base + 4u * head_v_dim + vd] = r1.x;
    recurrent_state[recurrent_head_base + 5u * head_v_dim + vd] = r1.y;
    recurrent_state[recurrent_head_base + 6u * head_v_dim + vd] = r1.z;
    recurrent_state[recurrent_head_base + 7u * head_v_dim + vd] = r1.w;
    recurrent_state[recurrent_head_base + 8u * head_v_dim + vd] = r2.x;
    recurrent_state[recurrent_head_base + 9u * head_v_dim + vd] = r2.y;
    recurrent_state[recurrent_head_base + 10u * head_v_dim + vd] = r2.z;
    recurrent_state[recurrent_head_base + 11u * head_v_dim + vd] = r2.w;
    recurrent_state[recurrent_head_base + 12u * head_v_dim + vd] = r3.x;
    recurrent_state[recurrent_head_base + 13u * head_v_dim + vd] = r3.y;
    recurrent_state[recurrent_head_base + 14u * head_v_dim + vd] = r3.z;
    recurrent_state[recurrent_head_base + 15u * head_v_dim + vd] = r3.w;
    recurrent_state[recurrent_head_base + 16u * head_v_dim + vd] = r4.x;
    recurrent_state[recurrent_head_base + 17u * head_v_dim + vd] = r4.y;
    recurrent_state[recurrent_head_base + 18u * head_v_dim + vd] = r4.z;
    recurrent_state[recurrent_head_base + 19u * head_v_dim + vd] = r4.w;
    recurrent_state[recurrent_head_base + 20u * head_v_dim + vd] = r5.x;
    recurrent_state[recurrent_head_base + 21u * head_v_dim + vd] = r5.y;
    recurrent_state[recurrent_head_base + 22u * head_v_dim + vd] = r5.z;
    recurrent_state[recurrent_head_base + 23u * head_v_dim + vd] = r5.w;
    recurrent_state[recurrent_head_base + 24u * head_v_dim + vd] = r6.x;
    recurrent_state[recurrent_head_base + 25u * head_v_dim + vd] = r6.y;
    recurrent_state[recurrent_head_base + 26u * head_v_dim + vd] = r6.z;
    recurrent_state[recurrent_head_base + 27u * head_v_dim + vd] = r6.w;
    recurrent_state[recurrent_head_base + 28u * head_v_dim + vd] = r7.x;
    recurrent_state[recurrent_head_base + 29u * head_v_dim + vd] = r7.y;
    recurrent_state[recurrent_head_base + 30u * head_v_dim + vd] = r7.z;
    recurrent_state[recurrent_head_base + 31u * head_v_dim + vd] = r7.w;
    recurrent_state[recurrent_head_base + 32u * head_v_dim + vd] = r8.x;
    recurrent_state[recurrent_head_base + 33u * head_v_dim + vd] = r8.y;
    recurrent_state[recurrent_head_base + 34u * head_v_dim + vd] = r8.z;
    recurrent_state[recurrent_head_base + 35u * head_v_dim + vd] = r8.w;
    recurrent_state[recurrent_head_base + 36u * head_v_dim + vd] = r9.x;
    recurrent_state[recurrent_head_base + 37u * head_v_dim + vd] = r9.y;
    recurrent_state[recurrent_head_base + 38u * head_v_dim + vd] = r9.z;
    recurrent_state[recurrent_head_base + 39u * head_v_dim + vd] = r9.w;
    recurrent_state[recurrent_head_base + 40u * head_v_dim + vd] = r10.x;
    recurrent_state[recurrent_head_base + 41u * head_v_dim + vd] = r10.y;
    recurrent_state[recurrent_head_base + 42u * head_v_dim + vd] = r10.z;
    recurrent_state[recurrent_head_base + 43u * head_v_dim + vd] = r10.w;
    recurrent_state[recurrent_head_base + 44u * head_v_dim + vd] = r11.x;
    recurrent_state[recurrent_head_base + 45u * head_v_dim + vd] = r11.y;
    recurrent_state[recurrent_head_base + 46u * head_v_dim + vd] = r11.z;
    recurrent_state[recurrent_head_base + 47u * head_v_dim + vd] = r11.w;
    recurrent_state[recurrent_head_base + 48u * head_v_dim + vd] = r12.x;
    recurrent_state[recurrent_head_base + 49u * head_v_dim + vd] = r12.y;
    recurrent_state[recurrent_head_base + 50u * head_v_dim + vd] = r12.z;
    recurrent_state[recurrent_head_base + 51u * head_v_dim + vd] = r12.w;
    recurrent_state[recurrent_head_base + 52u * head_v_dim + vd] = r13.x;
    recurrent_state[recurrent_head_base + 53u * head_v_dim + vd] = r13.y;
    recurrent_state[recurrent_head_base + 54u * head_v_dim + vd] = r13.z;
    recurrent_state[recurrent_head_base + 55u * head_v_dim + vd] = r13.w;
    recurrent_state[recurrent_head_base + 56u * head_v_dim + vd] = r14.x;
    recurrent_state[recurrent_head_base + 57u * head_v_dim + vd] = r14.y;
    recurrent_state[recurrent_head_base + 58u * head_v_dim + vd] = r14.z;
    recurrent_state[recurrent_head_base + 59u * head_v_dim + vd] = r14.w;
    recurrent_state[recurrent_head_base + 60u * head_v_dim + vd] = r15.x;
    recurrent_state[recurrent_head_base + 61u * head_v_dim + vd] = r15.y;
    recurrent_state[recurrent_head_base + 62u * head_v_dim + vd] = r15.z;
    recurrent_state[recurrent_head_base + 63u * head_v_dim + vd] = r15.w;
    recurrent_state[recurrent_head_base + 64u * head_v_dim + vd] = r16.x;
    recurrent_state[recurrent_head_base + 65u * head_v_dim + vd] = r16.y;
    recurrent_state[recurrent_head_base + 66u * head_v_dim + vd] = r16.z;
    recurrent_state[recurrent_head_base + 67u * head_v_dim + vd] = r16.w;
    recurrent_state[recurrent_head_base + 68u * head_v_dim + vd] = r17.x;
    recurrent_state[recurrent_head_base + 69u * head_v_dim + vd] = r17.y;
    recurrent_state[recurrent_head_base + 70u * head_v_dim + vd] = r17.z;
    recurrent_state[recurrent_head_base + 71u * head_v_dim + vd] = r17.w;
    recurrent_state[recurrent_head_base + 72u * head_v_dim + vd] = r18.x;
    recurrent_state[recurrent_head_base + 73u * head_v_dim + vd] = r18.y;
    recurrent_state[recurrent_head_base + 74u * head_v_dim + vd] = r18.z;
    recurrent_state[recurrent_head_base + 75u * head_v_dim + vd] = r18.w;
    recurrent_state[recurrent_head_base + 76u * head_v_dim + vd] = r19.x;
    recurrent_state[recurrent_head_base + 77u * head_v_dim + vd] = r19.y;
    recurrent_state[recurrent_head_base + 78u * head_v_dim + vd] = r19.z;
    recurrent_state[recurrent_head_base + 79u * head_v_dim + vd] = r19.w;
    recurrent_state[recurrent_head_base + 80u * head_v_dim + vd] = r20.x;
    recurrent_state[recurrent_head_base + 81u * head_v_dim + vd] = r20.y;
    recurrent_state[recurrent_head_base + 82u * head_v_dim + vd] = r20.z;
    recurrent_state[recurrent_head_base + 83u * head_v_dim + vd] = r20.w;
    recurrent_state[recurrent_head_base + 84u * head_v_dim + vd] = r21.x;
    recurrent_state[recurrent_head_base + 85u * head_v_dim + vd] = r21.y;
    recurrent_state[recurrent_head_base + 86u * head_v_dim + vd] = r21.z;
    recurrent_state[recurrent_head_base + 87u * head_v_dim + vd] = r21.w;
    recurrent_state[recurrent_head_base + 88u * head_v_dim + vd] = r22.x;
    recurrent_state[recurrent_head_base + 89u * head_v_dim + vd] = r22.y;
    recurrent_state[recurrent_head_base + 90u * head_v_dim + vd] = r22.z;
    recurrent_state[recurrent_head_base + 91u * head_v_dim + vd] = r22.w;
    recurrent_state[recurrent_head_base + 92u * head_v_dim + vd] = r23.x;
    recurrent_state[recurrent_head_base + 93u * head_v_dim + vd] = r23.y;
    recurrent_state[recurrent_head_base + 94u * head_v_dim + vd] = r23.z;
    recurrent_state[recurrent_head_base + 95u * head_v_dim + vd] = r23.w;
    recurrent_state[recurrent_head_base + 96u * head_v_dim + vd] = r24.x;
    recurrent_state[recurrent_head_base + 97u * head_v_dim + vd] = r24.y;
    recurrent_state[recurrent_head_base + 98u * head_v_dim + vd] = r24.z;
    recurrent_state[recurrent_head_base + 99u * head_v_dim + vd] = r24.w;
    recurrent_state[recurrent_head_base + 100u * head_v_dim + vd] = r25.x;
    recurrent_state[recurrent_head_base + 101u * head_v_dim + vd] = r25.y;
    recurrent_state[recurrent_head_base + 102u * head_v_dim + vd] = r25.z;
    recurrent_state[recurrent_head_base + 103u * head_v_dim + vd] = r25.w;
    recurrent_state[recurrent_head_base + 104u * head_v_dim + vd] = r26.x;
    recurrent_state[recurrent_head_base + 105u * head_v_dim + vd] = r26.y;
    recurrent_state[recurrent_head_base + 106u * head_v_dim + vd] = r26.z;
    recurrent_state[recurrent_head_base + 107u * head_v_dim + vd] = r26.w;
    recurrent_state[recurrent_head_base + 108u * head_v_dim + vd] = r27.x;
    recurrent_state[recurrent_head_base + 109u * head_v_dim + vd] = r27.y;
    recurrent_state[recurrent_head_base + 110u * head_v_dim + vd] = r27.z;
    recurrent_state[recurrent_head_base + 111u * head_v_dim + vd] = r27.w;
    recurrent_state[recurrent_head_base + 112u * head_v_dim + vd] = r28.x;
    recurrent_state[recurrent_head_base + 113u * head_v_dim + vd] = r28.y;
    recurrent_state[recurrent_head_base + 114u * head_v_dim + vd] = r28.z;
    recurrent_state[recurrent_head_base + 115u * head_v_dim + vd] = r28.w;
    recurrent_state[recurrent_head_base + 116u * head_v_dim + vd] = r29.x;
    recurrent_state[recurrent_head_base + 117u * head_v_dim + vd] = r29.y;
    recurrent_state[recurrent_head_base + 118u * head_v_dim + vd] = r29.z;
    recurrent_state[recurrent_head_base + 119u * head_v_dim + vd] = r29.w;
    recurrent_state[recurrent_head_base + 120u * head_v_dim + vd] = r30.x;
    recurrent_state[recurrent_head_base + 121u * head_v_dim + vd] = r30.y;
    recurrent_state[recurrent_head_base + 122u * head_v_dim + vd] = r30.z;
    recurrent_state[recurrent_head_base + 123u * head_v_dim + vd] = r30.w;
    recurrent_state[recurrent_head_base + 124u * head_v_dim + vd] = r31.x;
    recurrent_state[recurrent_head_base + 125u * head_v_dim + vd] = r31.y;
    recurrent_state[recurrent_head_base + 126u * head_v_dim + vd] = r31.z;
    recurrent_state[recurrent_head_base + 127u * head_v_dim + vd] = r31.w;
  }
}
