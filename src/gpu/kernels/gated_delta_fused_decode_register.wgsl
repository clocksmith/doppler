// AUTO-GENERATED from src/gpu/kernels/gated_delta_fused_decode.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
override RECURRENT_STATE_COMPONENTS: u32 = 1u;

override WORKGROUP_SIZE: u32 = 128u;

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
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> z_proj: array<f32>;
@group(0) @binding(3) var<storage, read> ab_proj: array<f32>;
@group(0) @binding(4) var<storage, read> conv_weight: array<f32>;
@group(0) @binding(5) var<storage, read_write> conv_state: array<f32>;
@group(0) @binding(6) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(7) var<storage, read> a_log: array<f32>;
@group(0) @binding(8) var<storage, read> norm_weight: array<f32>;
@group(0) @binding(9) var<storage, read_write> recurrent_state: array<f32>;
@group(0) @binding(10) var<storage, read_write> output: array<f32>;

var<workgroup> shared_q: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_k: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_reduce: array<f32, WORKGROUP_SIZE>;

fn softplus(x: f32) -> f32 {
  if (x > 20.0) {
    return x;
  }
  if (x < -20.0) {
    return exp_refined(x);
  }
  return log_refined(1.0 + exp_refined(x));
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


fn update_conv_channel(channel: u32) -> f32 {
  let kernel_size = params.conv_kernel_size;
  let state_base = channel * kernel_size;
  let newest = f32(qkv[channel]);

  for (var k: u32 = 0u; k + 1u < kernel_size; k = k + 1u) {
    conv_state[state_base + k] = conv_state[state_base + k + 1u];
  }
  conv_state[state_base + kernel_size - 1u] = newest;

  var mixed: f32 = 0.0;
  for (var k: u32 = 0u; k < kernel_size; k = k + 1u) {
    mixed = mixed + conv_state[state_base + k] * conv_weight[state_base + k];
  }
  return silu(mixed);
}

fn reduce_sum(value: f32, lid: u32) -> f32 {
  shared_reduce[lid] = value;
  workgroupBarrier();
  for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lid < stride) {
      shared_reduce[lid] = shared_reduce[lid] + shared_reduce[lid + stride];
    }
    workgroupBarrier();
  }
  let result = shared_reduce[0];
  // The next reduction may overwrite shared_reduce as soon as we return.
  workgroupBarrier();
  return result;
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
  let lane = lid.x;
  if (head >= params.num_v_heads || params.num_tokens != 1u || params.q_rep != 1u || params.head_k_dim != 128u || RECURRENT_STATE_COMPONENTS != 1u) {
    return;
  }

  let head_k_dim = params.head_k_dim;
  let head_v_dim = params.head_v_dim;
  let q_base = head * head_k_dim;
  let k_base = params.q_size + head * head_k_dim;
  let v_base = params.q_size + params.k_size + head * head_v_dim;
  let out_row_base = head * head_v_dim;
  let recurrent_head_base = head * head_k_dim * head_v_dim;
  let is_k_active = lane < head_k_dim;
  let is_v_active = lane < head_v_dim;

  var q_val = 0.0;
  var k_val = 0.0;
  var v_val = 0.0;
  if (is_k_active) {
    q_val = update_conv_channel(q_base + lane);
    k_val = update_conv_channel(k_base + lane);
  }
  if (is_v_active) {
    v_val = update_conv_channel(v_base + lane);
  }
  shared_q[lane] = select(0.0, q_val, is_k_active);
  shared_k[lane] = select(0.0, k_val, is_k_active);
  workgroupBarrier();

  let q_norm_sq = reduce_sum(select(0.0, q_val * q_val, is_k_active), lane);
  let head_scale = inverseSqrt(f32(head_k_dim));
  let q_norm_scale = head_scale * inverse_root_refined(q_norm_sq + params.qk_l2norm_eps);
  let k_norm_sq = reduce_sum(select(0.0, k_val * k_val, is_k_active), lane);
  let k_norm_scale = inverse_root_refined(k_norm_sq + params.qk_l2norm_eps);

  let ab_row_base = head;
  let b_index = params.b_proj_offset_elements + ab_row_base;
  let beta = sigmoid_refined(f32(ab_proj[b_index]));
  let g = -exp_refined(a_log[head]) * softplus(f32(ab_proj[ab_row_base]) + dt_bias[head]);
  let g_exp = exp_refined(g);

  var kv_mem = 0.0;
  var out_value = 0.0;
  if (is_v_active) {
      // Registered K=128 decode: keep each decayed scalar through its update.
      var state0 = vec4<f32>(recurrent_state[recurrent_head_base + 0u * head_v_dim + lane], recurrent_state[recurrent_head_base + 1u * head_v_dim + lane], recurrent_state[recurrent_head_base + 2u * head_v_dim + lane], recurrent_state[recurrent_head_base + 3u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state0.x, multiply_ordered(shared_k[0u], k_norm_scale), kv_mem);
      kv_mem = fma(state0.y, multiply_ordered(shared_k[1u], k_norm_scale), kv_mem);
      kv_mem = fma(state0.z, multiply_ordered(shared_k[2u], k_norm_scale), kv_mem);
      kv_mem = fma(state0.w, multiply_ordered(shared_k[3u], k_norm_scale), kv_mem);
      var state1 = vec4<f32>(recurrent_state[recurrent_head_base + 4u * head_v_dim + lane], recurrent_state[recurrent_head_base + 5u * head_v_dim + lane], recurrent_state[recurrent_head_base + 6u * head_v_dim + lane], recurrent_state[recurrent_head_base + 7u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state1.x, multiply_ordered(shared_k[4u], k_norm_scale), kv_mem);
      kv_mem = fma(state1.y, multiply_ordered(shared_k[5u], k_norm_scale), kv_mem);
      kv_mem = fma(state1.z, multiply_ordered(shared_k[6u], k_norm_scale), kv_mem);
      kv_mem = fma(state1.w, multiply_ordered(shared_k[7u], k_norm_scale), kv_mem);
      var state2 = vec4<f32>(recurrent_state[recurrent_head_base + 8u * head_v_dim + lane], recurrent_state[recurrent_head_base + 9u * head_v_dim + lane], recurrent_state[recurrent_head_base + 10u * head_v_dim + lane], recurrent_state[recurrent_head_base + 11u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state2.x, multiply_ordered(shared_k[8u], k_norm_scale), kv_mem);
      kv_mem = fma(state2.y, multiply_ordered(shared_k[9u], k_norm_scale), kv_mem);
      kv_mem = fma(state2.z, multiply_ordered(shared_k[10u], k_norm_scale), kv_mem);
      kv_mem = fma(state2.w, multiply_ordered(shared_k[11u], k_norm_scale), kv_mem);
      var state3 = vec4<f32>(recurrent_state[recurrent_head_base + 12u * head_v_dim + lane], recurrent_state[recurrent_head_base + 13u * head_v_dim + lane], recurrent_state[recurrent_head_base + 14u * head_v_dim + lane], recurrent_state[recurrent_head_base + 15u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state3.x, multiply_ordered(shared_k[12u], k_norm_scale), kv_mem);
      kv_mem = fma(state3.y, multiply_ordered(shared_k[13u], k_norm_scale), kv_mem);
      kv_mem = fma(state3.z, multiply_ordered(shared_k[14u], k_norm_scale), kv_mem);
      kv_mem = fma(state3.w, multiply_ordered(shared_k[15u], k_norm_scale), kv_mem);
      var state4 = vec4<f32>(recurrent_state[recurrent_head_base + 16u * head_v_dim + lane], recurrent_state[recurrent_head_base + 17u * head_v_dim + lane], recurrent_state[recurrent_head_base + 18u * head_v_dim + lane], recurrent_state[recurrent_head_base + 19u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state4.x, multiply_ordered(shared_k[16u], k_norm_scale), kv_mem);
      kv_mem = fma(state4.y, multiply_ordered(shared_k[17u], k_norm_scale), kv_mem);
      kv_mem = fma(state4.z, multiply_ordered(shared_k[18u], k_norm_scale), kv_mem);
      kv_mem = fma(state4.w, multiply_ordered(shared_k[19u], k_norm_scale), kv_mem);
      var state5 = vec4<f32>(recurrent_state[recurrent_head_base + 20u * head_v_dim + lane], recurrent_state[recurrent_head_base + 21u * head_v_dim + lane], recurrent_state[recurrent_head_base + 22u * head_v_dim + lane], recurrent_state[recurrent_head_base + 23u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state5.x, multiply_ordered(shared_k[20u], k_norm_scale), kv_mem);
      kv_mem = fma(state5.y, multiply_ordered(shared_k[21u], k_norm_scale), kv_mem);
      kv_mem = fma(state5.z, multiply_ordered(shared_k[22u], k_norm_scale), kv_mem);
      kv_mem = fma(state5.w, multiply_ordered(shared_k[23u], k_norm_scale), kv_mem);
      var state6 = vec4<f32>(recurrent_state[recurrent_head_base + 24u * head_v_dim + lane], recurrent_state[recurrent_head_base + 25u * head_v_dim + lane], recurrent_state[recurrent_head_base + 26u * head_v_dim + lane], recurrent_state[recurrent_head_base + 27u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state6.x, multiply_ordered(shared_k[24u], k_norm_scale), kv_mem);
      kv_mem = fma(state6.y, multiply_ordered(shared_k[25u], k_norm_scale), kv_mem);
      kv_mem = fma(state6.z, multiply_ordered(shared_k[26u], k_norm_scale), kv_mem);
      kv_mem = fma(state6.w, multiply_ordered(shared_k[27u], k_norm_scale), kv_mem);
      var state7 = vec4<f32>(recurrent_state[recurrent_head_base + 28u * head_v_dim + lane], recurrent_state[recurrent_head_base + 29u * head_v_dim + lane], recurrent_state[recurrent_head_base + 30u * head_v_dim + lane], recurrent_state[recurrent_head_base + 31u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state7.x, multiply_ordered(shared_k[28u], k_norm_scale), kv_mem);
      kv_mem = fma(state7.y, multiply_ordered(shared_k[29u], k_norm_scale), kv_mem);
      kv_mem = fma(state7.z, multiply_ordered(shared_k[30u], k_norm_scale), kv_mem);
      kv_mem = fma(state7.w, multiply_ordered(shared_k[31u], k_norm_scale), kv_mem);
      var state8 = vec4<f32>(recurrent_state[recurrent_head_base + 32u * head_v_dim + lane], recurrent_state[recurrent_head_base + 33u * head_v_dim + lane], recurrent_state[recurrent_head_base + 34u * head_v_dim + lane], recurrent_state[recurrent_head_base + 35u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state8.x, multiply_ordered(shared_k[32u], k_norm_scale), kv_mem);
      kv_mem = fma(state8.y, multiply_ordered(shared_k[33u], k_norm_scale), kv_mem);
      kv_mem = fma(state8.z, multiply_ordered(shared_k[34u], k_norm_scale), kv_mem);
      kv_mem = fma(state8.w, multiply_ordered(shared_k[35u], k_norm_scale), kv_mem);
      var state9 = vec4<f32>(recurrent_state[recurrent_head_base + 36u * head_v_dim + lane], recurrent_state[recurrent_head_base + 37u * head_v_dim + lane], recurrent_state[recurrent_head_base + 38u * head_v_dim + lane], recurrent_state[recurrent_head_base + 39u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state9.x, multiply_ordered(shared_k[36u], k_norm_scale), kv_mem);
      kv_mem = fma(state9.y, multiply_ordered(shared_k[37u], k_norm_scale), kv_mem);
      kv_mem = fma(state9.z, multiply_ordered(shared_k[38u], k_norm_scale), kv_mem);
      kv_mem = fma(state9.w, multiply_ordered(shared_k[39u], k_norm_scale), kv_mem);
      var state10 = vec4<f32>(recurrent_state[recurrent_head_base + 40u * head_v_dim + lane], recurrent_state[recurrent_head_base + 41u * head_v_dim + lane], recurrent_state[recurrent_head_base + 42u * head_v_dim + lane], recurrent_state[recurrent_head_base + 43u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state10.x, multiply_ordered(shared_k[40u], k_norm_scale), kv_mem);
      kv_mem = fma(state10.y, multiply_ordered(shared_k[41u], k_norm_scale), kv_mem);
      kv_mem = fma(state10.z, multiply_ordered(shared_k[42u], k_norm_scale), kv_mem);
      kv_mem = fma(state10.w, multiply_ordered(shared_k[43u], k_norm_scale), kv_mem);
      var state11 = vec4<f32>(recurrent_state[recurrent_head_base + 44u * head_v_dim + lane], recurrent_state[recurrent_head_base + 45u * head_v_dim + lane], recurrent_state[recurrent_head_base + 46u * head_v_dim + lane], recurrent_state[recurrent_head_base + 47u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state11.x, multiply_ordered(shared_k[44u], k_norm_scale), kv_mem);
      kv_mem = fma(state11.y, multiply_ordered(shared_k[45u], k_norm_scale), kv_mem);
      kv_mem = fma(state11.z, multiply_ordered(shared_k[46u], k_norm_scale), kv_mem);
      kv_mem = fma(state11.w, multiply_ordered(shared_k[47u], k_norm_scale), kv_mem);
      var state12 = vec4<f32>(recurrent_state[recurrent_head_base + 48u * head_v_dim + lane], recurrent_state[recurrent_head_base + 49u * head_v_dim + lane], recurrent_state[recurrent_head_base + 50u * head_v_dim + lane], recurrent_state[recurrent_head_base + 51u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state12.x, multiply_ordered(shared_k[48u], k_norm_scale), kv_mem);
      kv_mem = fma(state12.y, multiply_ordered(shared_k[49u], k_norm_scale), kv_mem);
      kv_mem = fma(state12.z, multiply_ordered(shared_k[50u], k_norm_scale), kv_mem);
      kv_mem = fma(state12.w, multiply_ordered(shared_k[51u], k_norm_scale), kv_mem);
      var state13 = vec4<f32>(recurrent_state[recurrent_head_base + 52u * head_v_dim + lane], recurrent_state[recurrent_head_base + 53u * head_v_dim + lane], recurrent_state[recurrent_head_base + 54u * head_v_dim + lane], recurrent_state[recurrent_head_base + 55u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state13.x, multiply_ordered(shared_k[52u], k_norm_scale), kv_mem);
      kv_mem = fma(state13.y, multiply_ordered(shared_k[53u], k_norm_scale), kv_mem);
      kv_mem = fma(state13.z, multiply_ordered(shared_k[54u], k_norm_scale), kv_mem);
      kv_mem = fma(state13.w, multiply_ordered(shared_k[55u], k_norm_scale), kv_mem);
      var state14 = vec4<f32>(recurrent_state[recurrent_head_base + 56u * head_v_dim + lane], recurrent_state[recurrent_head_base + 57u * head_v_dim + lane], recurrent_state[recurrent_head_base + 58u * head_v_dim + lane], recurrent_state[recurrent_head_base + 59u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state14.x, multiply_ordered(shared_k[56u], k_norm_scale), kv_mem);
      kv_mem = fma(state14.y, multiply_ordered(shared_k[57u], k_norm_scale), kv_mem);
      kv_mem = fma(state14.z, multiply_ordered(shared_k[58u], k_norm_scale), kv_mem);
      kv_mem = fma(state14.w, multiply_ordered(shared_k[59u], k_norm_scale), kv_mem);
      var state15 = vec4<f32>(recurrent_state[recurrent_head_base + 60u * head_v_dim + lane], recurrent_state[recurrent_head_base + 61u * head_v_dim + lane], recurrent_state[recurrent_head_base + 62u * head_v_dim + lane], recurrent_state[recurrent_head_base + 63u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state15.x, multiply_ordered(shared_k[60u], k_norm_scale), kv_mem);
      kv_mem = fma(state15.y, multiply_ordered(shared_k[61u], k_norm_scale), kv_mem);
      kv_mem = fma(state15.z, multiply_ordered(shared_k[62u], k_norm_scale), kv_mem);
      kv_mem = fma(state15.w, multiply_ordered(shared_k[63u], k_norm_scale), kv_mem);
      var state16 = vec4<f32>(recurrent_state[recurrent_head_base + 64u * head_v_dim + lane], recurrent_state[recurrent_head_base + 65u * head_v_dim + lane], recurrent_state[recurrent_head_base + 66u * head_v_dim + lane], recurrent_state[recurrent_head_base + 67u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state16.x, multiply_ordered(shared_k[64u], k_norm_scale), kv_mem);
      kv_mem = fma(state16.y, multiply_ordered(shared_k[65u], k_norm_scale), kv_mem);
      kv_mem = fma(state16.z, multiply_ordered(shared_k[66u], k_norm_scale), kv_mem);
      kv_mem = fma(state16.w, multiply_ordered(shared_k[67u], k_norm_scale), kv_mem);
      var state17 = vec4<f32>(recurrent_state[recurrent_head_base + 68u * head_v_dim + lane], recurrent_state[recurrent_head_base + 69u * head_v_dim + lane], recurrent_state[recurrent_head_base + 70u * head_v_dim + lane], recurrent_state[recurrent_head_base + 71u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state17.x, multiply_ordered(shared_k[68u], k_norm_scale), kv_mem);
      kv_mem = fma(state17.y, multiply_ordered(shared_k[69u], k_norm_scale), kv_mem);
      kv_mem = fma(state17.z, multiply_ordered(shared_k[70u], k_norm_scale), kv_mem);
      kv_mem = fma(state17.w, multiply_ordered(shared_k[71u], k_norm_scale), kv_mem);
      var state18 = vec4<f32>(recurrent_state[recurrent_head_base + 72u * head_v_dim + lane], recurrent_state[recurrent_head_base + 73u * head_v_dim + lane], recurrent_state[recurrent_head_base + 74u * head_v_dim + lane], recurrent_state[recurrent_head_base + 75u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state18.x, multiply_ordered(shared_k[72u], k_norm_scale), kv_mem);
      kv_mem = fma(state18.y, multiply_ordered(shared_k[73u], k_norm_scale), kv_mem);
      kv_mem = fma(state18.z, multiply_ordered(shared_k[74u], k_norm_scale), kv_mem);
      kv_mem = fma(state18.w, multiply_ordered(shared_k[75u], k_norm_scale), kv_mem);
      var state19 = vec4<f32>(recurrent_state[recurrent_head_base + 76u * head_v_dim + lane], recurrent_state[recurrent_head_base + 77u * head_v_dim + lane], recurrent_state[recurrent_head_base + 78u * head_v_dim + lane], recurrent_state[recurrent_head_base + 79u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state19.x, multiply_ordered(shared_k[76u], k_norm_scale), kv_mem);
      kv_mem = fma(state19.y, multiply_ordered(shared_k[77u], k_norm_scale), kv_mem);
      kv_mem = fma(state19.z, multiply_ordered(shared_k[78u], k_norm_scale), kv_mem);
      kv_mem = fma(state19.w, multiply_ordered(shared_k[79u], k_norm_scale), kv_mem);
      var state20 = vec4<f32>(recurrent_state[recurrent_head_base + 80u * head_v_dim + lane], recurrent_state[recurrent_head_base + 81u * head_v_dim + lane], recurrent_state[recurrent_head_base + 82u * head_v_dim + lane], recurrent_state[recurrent_head_base + 83u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state20.x, multiply_ordered(shared_k[80u], k_norm_scale), kv_mem);
      kv_mem = fma(state20.y, multiply_ordered(shared_k[81u], k_norm_scale), kv_mem);
      kv_mem = fma(state20.z, multiply_ordered(shared_k[82u], k_norm_scale), kv_mem);
      kv_mem = fma(state20.w, multiply_ordered(shared_k[83u], k_norm_scale), kv_mem);
      var state21 = vec4<f32>(recurrent_state[recurrent_head_base + 84u * head_v_dim + lane], recurrent_state[recurrent_head_base + 85u * head_v_dim + lane], recurrent_state[recurrent_head_base + 86u * head_v_dim + lane], recurrent_state[recurrent_head_base + 87u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state21.x, multiply_ordered(shared_k[84u], k_norm_scale), kv_mem);
      kv_mem = fma(state21.y, multiply_ordered(shared_k[85u], k_norm_scale), kv_mem);
      kv_mem = fma(state21.z, multiply_ordered(shared_k[86u], k_norm_scale), kv_mem);
      kv_mem = fma(state21.w, multiply_ordered(shared_k[87u], k_norm_scale), kv_mem);
      var state22 = vec4<f32>(recurrent_state[recurrent_head_base + 88u * head_v_dim + lane], recurrent_state[recurrent_head_base + 89u * head_v_dim + lane], recurrent_state[recurrent_head_base + 90u * head_v_dim + lane], recurrent_state[recurrent_head_base + 91u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state22.x, multiply_ordered(shared_k[88u], k_norm_scale), kv_mem);
      kv_mem = fma(state22.y, multiply_ordered(shared_k[89u], k_norm_scale), kv_mem);
      kv_mem = fma(state22.z, multiply_ordered(shared_k[90u], k_norm_scale), kv_mem);
      kv_mem = fma(state22.w, multiply_ordered(shared_k[91u], k_norm_scale), kv_mem);
      var state23 = vec4<f32>(recurrent_state[recurrent_head_base + 92u * head_v_dim + lane], recurrent_state[recurrent_head_base + 93u * head_v_dim + lane], recurrent_state[recurrent_head_base + 94u * head_v_dim + lane], recurrent_state[recurrent_head_base + 95u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state23.x, multiply_ordered(shared_k[92u], k_norm_scale), kv_mem);
      kv_mem = fma(state23.y, multiply_ordered(shared_k[93u], k_norm_scale), kv_mem);
      kv_mem = fma(state23.z, multiply_ordered(shared_k[94u], k_norm_scale), kv_mem);
      kv_mem = fma(state23.w, multiply_ordered(shared_k[95u], k_norm_scale), kv_mem);
      var state24 = vec4<f32>(recurrent_state[recurrent_head_base + 96u * head_v_dim + lane], recurrent_state[recurrent_head_base + 97u * head_v_dim + lane], recurrent_state[recurrent_head_base + 98u * head_v_dim + lane], recurrent_state[recurrent_head_base + 99u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state24.x, multiply_ordered(shared_k[96u], k_norm_scale), kv_mem);
      kv_mem = fma(state24.y, multiply_ordered(shared_k[97u], k_norm_scale), kv_mem);
      kv_mem = fma(state24.z, multiply_ordered(shared_k[98u], k_norm_scale), kv_mem);
      kv_mem = fma(state24.w, multiply_ordered(shared_k[99u], k_norm_scale), kv_mem);
      var state25 = vec4<f32>(recurrent_state[recurrent_head_base + 100u * head_v_dim + lane], recurrent_state[recurrent_head_base + 101u * head_v_dim + lane], recurrent_state[recurrent_head_base + 102u * head_v_dim + lane], recurrent_state[recurrent_head_base + 103u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state25.x, multiply_ordered(shared_k[100u], k_norm_scale), kv_mem);
      kv_mem = fma(state25.y, multiply_ordered(shared_k[101u], k_norm_scale), kv_mem);
      kv_mem = fma(state25.z, multiply_ordered(shared_k[102u], k_norm_scale), kv_mem);
      kv_mem = fma(state25.w, multiply_ordered(shared_k[103u], k_norm_scale), kv_mem);
      var state26 = vec4<f32>(recurrent_state[recurrent_head_base + 104u * head_v_dim + lane], recurrent_state[recurrent_head_base + 105u * head_v_dim + lane], recurrent_state[recurrent_head_base + 106u * head_v_dim + lane], recurrent_state[recurrent_head_base + 107u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state26.x, multiply_ordered(shared_k[104u], k_norm_scale), kv_mem);
      kv_mem = fma(state26.y, multiply_ordered(shared_k[105u], k_norm_scale), kv_mem);
      kv_mem = fma(state26.z, multiply_ordered(shared_k[106u], k_norm_scale), kv_mem);
      kv_mem = fma(state26.w, multiply_ordered(shared_k[107u], k_norm_scale), kv_mem);
      var state27 = vec4<f32>(recurrent_state[recurrent_head_base + 108u * head_v_dim + lane], recurrent_state[recurrent_head_base + 109u * head_v_dim + lane], recurrent_state[recurrent_head_base + 110u * head_v_dim + lane], recurrent_state[recurrent_head_base + 111u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state27.x, multiply_ordered(shared_k[108u], k_norm_scale), kv_mem);
      kv_mem = fma(state27.y, multiply_ordered(shared_k[109u], k_norm_scale), kv_mem);
      kv_mem = fma(state27.z, multiply_ordered(shared_k[110u], k_norm_scale), kv_mem);
      kv_mem = fma(state27.w, multiply_ordered(shared_k[111u], k_norm_scale), kv_mem);
      var state28 = vec4<f32>(recurrent_state[recurrent_head_base + 112u * head_v_dim + lane], recurrent_state[recurrent_head_base + 113u * head_v_dim + lane], recurrent_state[recurrent_head_base + 114u * head_v_dim + lane], recurrent_state[recurrent_head_base + 115u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state28.x, multiply_ordered(shared_k[112u], k_norm_scale), kv_mem);
      kv_mem = fma(state28.y, multiply_ordered(shared_k[113u], k_norm_scale), kv_mem);
      kv_mem = fma(state28.z, multiply_ordered(shared_k[114u], k_norm_scale), kv_mem);
      kv_mem = fma(state28.w, multiply_ordered(shared_k[115u], k_norm_scale), kv_mem);
      var state29 = vec4<f32>(recurrent_state[recurrent_head_base + 116u * head_v_dim + lane], recurrent_state[recurrent_head_base + 117u * head_v_dim + lane], recurrent_state[recurrent_head_base + 118u * head_v_dim + lane], recurrent_state[recurrent_head_base + 119u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state29.x, multiply_ordered(shared_k[116u], k_norm_scale), kv_mem);
      kv_mem = fma(state29.y, multiply_ordered(shared_k[117u], k_norm_scale), kv_mem);
      kv_mem = fma(state29.z, multiply_ordered(shared_k[118u], k_norm_scale), kv_mem);
      kv_mem = fma(state29.w, multiply_ordered(shared_k[119u], k_norm_scale), kv_mem);
      var state30 = vec4<f32>(recurrent_state[recurrent_head_base + 120u * head_v_dim + lane], recurrent_state[recurrent_head_base + 121u * head_v_dim + lane], recurrent_state[recurrent_head_base + 122u * head_v_dim + lane], recurrent_state[recurrent_head_base + 123u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state30.x, multiply_ordered(shared_k[120u], k_norm_scale), kv_mem);
      kv_mem = fma(state30.y, multiply_ordered(shared_k[121u], k_norm_scale), kv_mem);
      kv_mem = fma(state30.z, multiply_ordered(shared_k[122u], k_norm_scale), kv_mem);
      kv_mem = fma(state30.w, multiply_ordered(shared_k[123u], k_norm_scale), kv_mem);
      var state31 = vec4<f32>(recurrent_state[recurrent_head_base + 124u * head_v_dim + lane], recurrent_state[recurrent_head_base + 125u * head_v_dim + lane], recurrent_state[recurrent_head_base + 126u * head_v_dim + lane], recurrent_state[recurrent_head_base + 127u * head_v_dim + lane]) * g_exp;
      kv_mem = fma(state31.x, multiply_ordered(shared_k[124u], k_norm_scale), kv_mem);
      kv_mem = fma(state31.y, multiply_ordered(shared_k[125u], k_norm_scale), kv_mem);
      kv_mem = fma(state31.z, multiply_ordered(shared_k[126u], k_norm_scale), kv_mem);
      kv_mem = fma(state31.w, multiply_ordered(shared_k[127u], k_norm_scale), kv_mem);
      let delta = multiply_ordered(v_val - kv_mem, beta);
      state0 = fma(vec4<f32>(multiply_ordered(shared_k[0u], k_norm_scale), multiply_ordered(shared_k[1u], k_norm_scale), multiply_ordered(shared_k[2u], k_norm_scale), multiply_ordered(shared_k[3u], k_norm_scale)), vec4<f32>(delta), state0);
      recurrent_state[recurrent_head_base + 0u * head_v_dim + lane] = state0.x;
      out_value = fma(state0.x, fma(shared_q[0u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 1u * head_v_dim + lane] = state0.y;
      out_value = fma(state0.y, fma(shared_q[1u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 2u * head_v_dim + lane] = state0.z;
      out_value = fma(state0.z, fma(shared_q[2u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 3u * head_v_dim + lane] = state0.w;
      out_value = fma(state0.w, fma(shared_q[3u], q_norm_scale, 0.0), out_value);
      state1 = fma(vec4<f32>(multiply_ordered(shared_k[4u], k_norm_scale), multiply_ordered(shared_k[5u], k_norm_scale), multiply_ordered(shared_k[6u], k_norm_scale), multiply_ordered(shared_k[7u], k_norm_scale)), vec4<f32>(delta), state1);
      recurrent_state[recurrent_head_base + 4u * head_v_dim + lane] = state1.x;
      out_value = fma(state1.x, fma(shared_q[4u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 5u * head_v_dim + lane] = state1.y;
      out_value = fma(state1.y, fma(shared_q[5u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 6u * head_v_dim + lane] = state1.z;
      out_value = fma(state1.z, fma(shared_q[6u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 7u * head_v_dim + lane] = state1.w;
      out_value = fma(state1.w, fma(shared_q[7u], q_norm_scale, 0.0), out_value);
      state2 = fma(vec4<f32>(multiply_ordered(shared_k[8u], k_norm_scale), multiply_ordered(shared_k[9u], k_norm_scale), multiply_ordered(shared_k[10u], k_norm_scale), multiply_ordered(shared_k[11u], k_norm_scale)), vec4<f32>(delta), state2);
      recurrent_state[recurrent_head_base + 8u * head_v_dim + lane] = state2.x;
      out_value = fma(state2.x, fma(shared_q[8u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 9u * head_v_dim + lane] = state2.y;
      out_value = fma(state2.y, fma(shared_q[9u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 10u * head_v_dim + lane] = state2.z;
      out_value = fma(state2.z, fma(shared_q[10u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 11u * head_v_dim + lane] = state2.w;
      out_value = fma(state2.w, fma(shared_q[11u], q_norm_scale, 0.0), out_value);
      state3 = fma(vec4<f32>(multiply_ordered(shared_k[12u], k_norm_scale), multiply_ordered(shared_k[13u], k_norm_scale), multiply_ordered(shared_k[14u], k_norm_scale), multiply_ordered(shared_k[15u], k_norm_scale)), vec4<f32>(delta), state3);
      recurrent_state[recurrent_head_base + 12u * head_v_dim + lane] = state3.x;
      out_value = fma(state3.x, fma(shared_q[12u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 13u * head_v_dim + lane] = state3.y;
      out_value = fma(state3.y, fma(shared_q[13u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 14u * head_v_dim + lane] = state3.z;
      out_value = fma(state3.z, fma(shared_q[14u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 15u * head_v_dim + lane] = state3.w;
      out_value = fma(state3.w, fma(shared_q[15u], q_norm_scale, 0.0), out_value);
      state4 = fma(vec4<f32>(multiply_ordered(shared_k[16u], k_norm_scale), multiply_ordered(shared_k[17u], k_norm_scale), multiply_ordered(shared_k[18u], k_norm_scale), multiply_ordered(shared_k[19u], k_norm_scale)), vec4<f32>(delta), state4);
      recurrent_state[recurrent_head_base + 16u * head_v_dim + lane] = state4.x;
      out_value = fma(state4.x, fma(shared_q[16u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 17u * head_v_dim + lane] = state4.y;
      out_value = fma(state4.y, fma(shared_q[17u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 18u * head_v_dim + lane] = state4.z;
      out_value = fma(state4.z, fma(shared_q[18u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 19u * head_v_dim + lane] = state4.w;
      out_value = fma(state4.w, fma(shared_q[19u], q_norm_scale, 0.0), out_value);
      state5 = fma(vec4<f32>(multiply_ordered(shared_k[20u], k_norm_scale), multiply_ordered(shared_k[21u], k_norm_scale), multiply_ordered(shared_k[22u], k_norm_scale), multiply_ordered(shared_k[23u], k_norm_scale)), vec4<f32>(delta), state5);
      recurrent_state[recurrent_head_base + 20u * head_v_dim + lane] = state5.x;
      out_value = fma(state5.x, fma(shared_q[20u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 21u * head_v_dim + lane] = state5.y;
      out_value = fma(state5.y, fma(shared_q[21u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 22u * head_v_dim + lane] = state5.z;
      out_value = fma(state5.z, fma(shared_q[22u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 23u * head_v_dim + lane] = state5.w;
      out_value = fma(state5.w, fma(shared_q[23u], q_norm_scale, 0.0), out_value);
      state6 = fma(vec4<f32>(multiply_ordered(shared_k[24u], k_norm_scale), multiply_ordered(shared_k[25u], k_norm_scale), multiply_ordered(shared_k[26u], k_norm_scale), multiply_ordered(shared_k[27u], k_norm_scale)), vec4<f32>(delta), state6);
      recurrent_state[recurrent_head_base + 24u * head_v_dim + lane] = state6.x;
      out_value = fma(state6.x, fma(shared_q[24u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 25u * head_v_dim + lane] = state6.y;
      out_value = fma(state6.y, fma(shared_q[25u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 26u * head_v_dim + lane] = state6.z;
      out_value = fma(state6.z, fma(shared_q[26u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 27u * head_v_dim + lane] = state6.w;
      out_value = fma(state6.w, fma(shared_q[27u], q_norm_scale, 0.0), out_value);
      state7 = fma(vec4<f32>(multiply_ordered(shared_k[28u], k_norm_scale), multiply_ordered(shared_k[29u], k_norm_scale), multiply_ordered(shared_k[30u], k_norm_scale), multiply_ordered(shared_k[31u], k_norm_scale)), vec4<f32>(delta), state7);
      recurrent_state[recurrent_head_base + 28u * head_v_dim + lane] = state7.x;
      out_value = fma(state7.x, fma(shared_q[28u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 29u * head_v_dim + lane] = state7.y;
      out_value = fma(state7.y, fma(shared_q[29u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 30u * head_v_dim + lane] = state7.z;
      out_value = fma(state7.z, fma(shared_q[30u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 31u * head_v_dim + lane] = state7.w;
      out_value = fma(state7.w, fma(shared_q[31u], q_norm_scale, 0.0), out_value);
      state8 = fma(vec4<f32>(multiply_ordered(shared_k[32u], k_norm_scale), multiply_ordered(shared_k[33u], k_norm_scale), multiply_ordered(shared_k[34u], k_norm_scale), multiply_ordered(shared_k[35u], k_norm_scale)), vec4<f32>(delta), state8);
      recurrent_state[recurrent_head_base + 32u * head_v_dim + lane] = state8.x;
      out_value = fma(state8.x, fma(shared_q[32u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 33u * head_v_dim + lane] = state8.y;
      out_value = fma(state8.y, fma(shared_q[33u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 34u * head_v_dim + lane] = state8.z;
      out_value = fma(state8.z, fma(shared_q[34u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 35u * head_v_dim + lane] = state8.w;
      out_value = fma(state8.w, fma(shared_q[35u], q_norm_scale, 0.0), out_value);
      state9 = fma(vec4<f32>(multiply_ordered(shared_k[36u], k_norm_scale), multiply_ordered(shared_k[37u], k_norm_scale), multiply_ordered(shared_k[38u], k_norm_scale), multiply_ordered(shared_k[39u], k_norm_scale)), vec4<f32>(delta), state9);
      recurrent_state[recurrent_head_base + 36u * head_v_dim + lane] = state9.x;
      out_value = fma(state9.x, fma(shared_q[36u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 37u * head_v_dim + lane] = state9.y;
      out_value = fma(state9.y, fma(shared_q[37u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 38u * head_v_dim + lane] = state9.z;
      out_value = fma(state9.z, fma(shared_q[38u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 39u * head_v_dim + lane] = state9.w;
      out_value = fma(state9.w, fma(shared_q[39u], q_norm_scale, 0.0), out_value);
      state10 = fma(vec4<f32>(multiply_ordered(shared_k[40u], k_norm_scale), multiply_ordered(shared_k[41u], k_norm_scale), multiply_ordered(shared_k[42u], k_norm_scale), multiply_ordered(shared_k[43u], k_norm_scale)), vec4<f32>(delta), state10);
      recurrent_state[recurrent_head_base + 40u * head_v_dim + lane] = state10.x;
      out_value = fma(state10.x, fma(shared_q[40u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 41u * head_v_dim + lane] = state10.y;
      out_value = fma(state10.y, fma(shared_q[41u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 42u * head_v_dim + lane] = state10.z;
      out_value = fma(state10.z, fma(shared_q[42u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 43u * head_v_dim + lane] = state10.w;
      out_value = fma(state10.w, fma(shared_q[43u], q_norm_scale, 0.0), out_value);
      state11 = fma(vec4<f32>(multiply_ordered(shared_k[44u], k_norm_scale), multiply_ordered(shared_k[45u], k_norm_scale), multiply_ordered(shared_k[46u], k_norm_scale), multiply_ordered(shared_k[47u], k_norm_scale)), vec4<f32>(delta), state11);
      recurrent_state[recurrent_head_base + 44u * head_v_dim + lane] = state11.x;
      out_value = fma(state11.x, fma(shared_q[44u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 45u * head_v_dim + lane] = state11.y;
      out_value = fma(state11.y, fma(shared_q[45u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 46u * head_v_dim + lane] = state11.z;
      out_value = fma(state11.z, fma(shared_q[46u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 47u * head_v_dim + lane] = state11.w;
      out_value = fma(state11.w, fma(shared_q[47u], q_norm_scale, 0.0), out_value);
      state12 = fma(vec4<f32>(multiply_ordered(shared_k[48u], k_norm_scale), multiply_ordered(shared_k[49u], k_norm_scale), multiply_ordered(shared_k[50u], k_norm_scale), multiply_ordered(shared_k[51u], k_norm_scale)), vec4<f32>(delta), state12);
      recurrent_state[recurrent_head_base + 48u * head_v_dim + lane] = state12.x;
      out_value = fma(state12.x, fma(shared_q[48u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 49u * head_v_dim + lane] = state12.y;
      out_value = fma(state12.y, fma(shared_q[49u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 50u * head_v_dim + lane] = state12.z;
      out_value = fma(state12.z, fma(shared_q[50u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 51u * head_v_dim + lane] = state12.w;
      out_value = fma(state12.w, fma(shared_q[51u], q_norm_scale, 0.0), out_value);
      state13 = fma(vec4<f32>(multiply_ordered(shared_k[52u], k_norm_scale), multiply_ordered(shared_k[53u], k_norm_scale), multiply_ordered(shared_k[54u], k_norm_scale), multiply_ordered(shared_k[55u], k_norm_scale)), vec4<f32>(delta), state13);
      recurrent_state[recurrent_head_base + 52u * head_v_dim + lane] = state13.x;
      out_value = fma(state13.x, fma(shared_q[52u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 53u * head_v_dim + lane] = state13.y;
      out_value = fma(state13.y, fma(shared_q[53u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 54u * head_v_dim + lane] = state13.z;
      out_value = fma(state13.z, fma(shared_q[54u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 55u * head_v_dim + lane] = state13.w;
      out_value = fma(state13.w, fma(shared_q[55u], q_norm_scale, 0.0), out_value);
      state14 = fma(vec4<f32>(multiply_ordered(shared_k[56u], k_norm_scale), multiply_ordered(shared_k[57u], k_norm_scale), multiply_ordered(shared_k[58u], k_norm_scale), multiply_ordered(shared_k[59u], k_norm_scale)), vec4<f32>(delta), state14);
      recurrent_state[recurrent_head_base + 56u * head_v_dim + lane] = state14.x;
      out_value = fma(state14.x, fma(shared_q[56u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 57u * head_v_dim + lane] = state14.y;
      out_value = fma(state14.y, fma(shared_q[57u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 58u * head_v_dim + lane] = state14.z;
      out_value = fma(state14.z, fma(shared_q[58u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 59u * head_v_dim + lane] = state14.w;
      out_value = fma(state14.w, fma(shared_q[59u], q_norm_scale, 0.0), out_value);
      state15 = fma(vec4<f32>(multiply_ordered(shared_k[60u], k_norm_scale), multiply_ordered(shared_k[61u], k_norm_scale), multiply_ordered(shared_k[62u], k_norm_scale), multiply_ordered(shared_k[63u], k_norm_scale)), vec4<f32>(delta), state15);
      recurrent_state[recurrent_head_base + 60u * head_v_dim + lane] = state15.x;
      out_value = fma(state15.x, fma(shared_q[60u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 61u * head_v_dim + lane] = state15.y;
      out_value = fma(state15.y, fma(shared_q[61u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 62u * head_v_dim + lane] = state15.z;
      out_value = fma(state15.z, fma(shared_q[62u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 63u * head_v_dim + lane] = state15.w;
      out_value = fma(state15.w, fma(shared_q[63u], q_norm_scale, 0.0), out_value);
      state16 = fma(vec4<f32>(multiply_ordered(shared_k[64u], k_norm_scale), multiply_ordered(shared_k[65u], k_norm_scale), multiply_ordered(shared_k[66u], k_norm_scale), multiply_ordered(shared_k[67u], k_norm_scale)), vec4<f32>(delta), state16);
      recurrent_state[recurrent_head_base + 64u * head_v_dim + lane] = state16.x;
      out_value = fma(state16.x, fma(shared_q[64u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 65u * head_v_dim + lane] = state16.y;
      out_value = fma(state16.y, fma(shared_q[65u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 66u * head_v_dim + lane] = state16.z;
      out_value = fma(state16.z, fma(shared_q[66u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 67u * head_v_dim + lane] = state16.w;
      out_value = fma(state16.w, fma(shared_q[67u], q_norm_scale, 0.0), out_value);
      state17 = fma(vec4<f32>(multiply_ordered(shared_k[68u], k_norm_scale), multiply_ordered(shared_k[69u], k_norm_scale), multiply_ordered(shared_k[70u], k_norm_scale), multiply_ordered(shared_k[71u], k_norm_scale)), vec4<f32>(delta), state17);
      recurrent_state[recurrent_head_base + 68u * head_v_dim + lane] = state17.x;
      out_value = fma(state17.x, fma(shared_q[68u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 69u * head_v_dim + lane] = state17.y;
      out_value = fma(state17.y, fma(shared_q[69u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 70u * head_v_dim + lane] = state17.z;
      out_value = fma(state17.z, fma(shared_q[70u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 71u * head_v_dim + lane] = state17.w;
      out_value = fma(state17.w, fma(shared_q[71u], q_norm_scale, 0.0), out_value);
      state18 = fma(vec4<f32>(multiply_ordered(shared_k[72u], k_norm_scale), multiply_ordered(shared_k[73u], k_norm_scale), multiply_ordered(shared_k[74u], k_norm_scale), multiply_ordered(shared_k[75u], k_norm_scale)), vec4<f32>(delta), state18);
      recurrent_state[recurrent_head_base + 72u * head_v_dim + lane] = state18.x;
      out_value = fma(state18.x, fma(shared_q[72u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 73u * head_v_dim + lane] = state18.y;
      out_value = fma(state18.y, fma(shared_q[73u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 74u * head_v_dim + lane] = state18.z;
      out_value = fma(state18.z, fma(shared_q[74u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 75u * head_v_dim + lane] = state18.w;
      out_value = fma(state18.w, fma(shared_q[75u], q_norm_scale, 0.0), out_value);
      state19 = fma(vec4<f32>(multiply_ordered(shared_k[76u], k_norm_scale), multiply_ordered(shared_k[77u], k_norm_scale), multiply_ordered(shared_k[78u], k_norm_scale), multiply_ordered(shared_k[79u], k_norm_scale)), vec4<f32>(delta), state19);
      recurrent_state[recurrent_head_base + 76u * head_v_dim + lane] = state19.x;
      out_value = fma(state19.x, fma(shared_q[76u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 77u * head_v_dim + lane] = state19.y;
      out_value = fma(state19.y, fma(shared_q[77u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 78u * head_v_dim + lane] = state19.z;
      out_value = fma(state19.z, fma(shared_q[78u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 79u * head_v_dim + lane] = state19.w;
      out_value = fma(state19.w, fma(shared_q[79u], q_norm_scale, 0.0), out_value);
      state20 = fma(vec4<f32>(multiply_ordered(shared_k[80u], k_norm_scale), multiply_ordered(shared_k[81u], k_norm_scale), multiply_ordered(shared_k[82u], k_norm_scale), multiply_ordered(shared_k[83u], k_norm_scale)), vec4<f32>(delta), state20);
      recurrent_state[recurrent_head_base + 80u * head_v_dim + lane] = state20.x;
      out_value = fma(state20.x, fma(shared_q[80u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 81u * head_v_dim + lane] = state20.y;
      out_value = fma(state20.y, fma(shared_q[81u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 82u * head_v_dim + lane] = state20.z;
      out_value = fma(state20.z, fma(shared_q[82u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 83u * head_v_dim + lane] = state20.w;
      out_value = fma(state20.w, fma(shared_q[83u], q_norm_scale, 0.0), out_value);
      state21 = fma(vec4<f32>(multiply_ordered(shared_k[84u], k_norm_scale), multiply_ordered(shared_k[85u], k_norm_scale), multiply_ordered(shared_k[86u], k_norm_scale), multiply_ordered(shared_k[87u], k_norm_scale)), vec4<f32>(delta), state21);
      recurrent_state[recurrent_head_base + 84u * head_v_dim + lane] = state21.x;
      out_value = fma(state21.x, fma(shared_q[84u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 85u * head_v_dim + lane] = state21.y;
      out_value = fma(state21.y, fma(shared_q[85u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 86u * head_v_dim + lane] = state21.z;
      out_value = fma(state21.z, fma(shared_q[86u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 87u * head_v_dim + lane] = state21.w;
      out_value = fma(state21.w, fma(shared_q[87u], q_norm_scale, 0.0), out_value);
      state22 = fma(vec4<f32>(multiply_ordered(shared_k[88u], k_norm_scale), multiply_ordered(shared_k[89u], k_norm_scale), multiply_ordered(shared_k[90u], k_norm_scale), multiply_ordered(shared_k[91u], k_norm_scale)), vec4<f32>(delta), state22);
      recurrent_state[recurrent_head_base + 88u * head_v_dim + lane] = state22.x;
      out_value = fma(state22.x, fma(shared_q[88u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 89u * head_v_dim + lane] = state22.y;
      out_value = fma(state22.y, fma(shared_q[89u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 90u * head_v_dim + lane] = state22.z;
      out_value = fma(state22.z, fma(shared_q[90u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 91u * head_v_dim + lane] = state22.w;
      out_value = fma(state22.w, fma(shared_q[91u], q_norm_scale, 0.0), out_value);
      state23 = fma(vec4<f32>(multiply_ordered(shared_k[92u], k_norm_scale), multiply_ordered(shared_k[93u], k_norm_scale), multiply_ordered(shared_k[94u], k_norm_scale), multiply_ordered(shared_k[95u], k_norm_scale)), vec4<f32>(delta), state23);
      recurrent_state[recurrent_head_base + 92u * head_v_dim + lane] = state23.x;
      out_value = fma(state23.x, fma(shared_q[92u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 93u * head_v_dim + lane] = state23.y;
      out_value = fma(state23.y, fma(shared_q[93u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 94u * head_v_dim + lane] = state23.z;
      out_value = fma(state23.z, fma(shared_q[94u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 95u * head_v_dim + lane] = state23.w;
      out_value = fma(state23.w, fma(shared_q[95u], q_norm_scale, 0.0), out_value);
      state24 = fma(vec4<f32>(multiply_ordered(shared_k[96u], k_norm_scale), multiply_ordered(shared_k[97u], k_norm_scale), multiply_ordered(shared_k[98u], k_norm_scale), multiply_ordered(shared_k[99u], k_norm_scale)), vec4<f32>(delta), state24);
      recurrent_state[recurrent_head_base + 96u * head_v_dim + lane] = state24.x;
      out_value = fma(state24.x, fma(shared_q[96u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 97u * head_v_dim + lane] = state24.y;
      out_value = fma(state24.y, fma(shared_q[97u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 98u * head_v_dim + lane] = state24.z;
      out_value = fma(state24.z, fma(shared_q[98u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 99u * head_v_dim + lane] = state24.w;
      out_value = fma(state24.w, fma(shared_q[99u], q_norm_scale, 0.0), out_value);
      state25 = fma(vec4<f32>(multiply_ordered(shared_k[100u], k_norm_scale), multiply_ordered(shared_k[101u], k_norm_scale), multiply_ordered(shared_k[102u], k_norm_scale), multiply_ordered(shared_k[103u], k_norm_scale)), vec4<f32>(delta), state25);
      recurrent_state[recurrent_head_base + 100u * head_v_dim + lane] = state25.x;
      out_value = fma(state25.x, fma(shared_q[100u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 101u * head_v_dim + lane] = state25.y;
      out_value = fma(state25.y, fma(shared_q[101u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 102u * head_v_dim + lane] = state25.z;
      out_value = fma(state25.z, fma(shared_q[102u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 103u * head_v_dim + lane] = state25.w;
      out_value = fma(state25.w, fma(shared_q[103u], q_norm_scale, 0.0), out_value);
      state26 = fma(vec4<f32>(multiply_ordered(shared_k[104u], k_norm_scale), multiply_ordered(shared_k[105u], k_norm_scale), multiply_ordered(shared_k[106u], k_norm_scale), multiply_ordered(shared_k[107u], k_norm_scale)), vec4<f32>(delta), state26);
      recurrent_state[recurrent_head_base + 104u * head_v_dim + lane] = state26.x;
      out_value = fma(state26.x, fma(shared_q[104u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 105u * head_v_dim + lane] = state26.y;
      out_value = fma(state26.y, fma(shared_q[105u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 106u * head_v_dim + lane] = state26.z;
      out_value = fma(state26.z, fma(shared_q[106u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 107u * head_v_dim + lane] = state26.w;
      out_value = fma(state26.w, fma(shared_q[107u], q_norm_scale, 0.0), out_value);
      state27 = fma(vec4<f32>(multiply_ordered(shared_k[108u], k_norm_scale), multiply_ordered(shared_k[109u], k_norm_scale), multiply_ordered(shared_k[110u], k_norm_scale), multiply_ordered(shared_k[111u], k_norm_scale)), vec4<f32>(delta), state27);
      recurrent_state[recurrent_head_base + 108u * head_v_dim + lane] = state27.x;
      out_value = fma(state27.x, fma(shared_q[108u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 109u * head_v_dim + lane] = state27.y;
      out_value = fma(state27.y, fma(shared_q[109u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 110u * head_v_dim + lane] = state27.z;
      out_value = fma(state27.z, fma(shared_q[110u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 111u * head_v_dim + lane] = state27.w;
      out_value = fma(state27.w, fma(shared_q[111u], q_norm_scale, 0.0), out_value);
      state28 = fma(vec4<f32>(multiply_ordered(shared_k[112u], k_norm_scale), multiply_ordered(shared_k[113u], k_norm_scale), multiply_ordered(shared_k[114u], k_norm_scale), multiply_ordered(shared_k[115u], k_norm_scale)), vec4<f32>(delta), state28);
      recurrent_state[recurrent_head_base + 112u * head_v_dim + lane] = state28.x;
      out_value = fma(state28.x, fma(shared_q[112u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 113u * head_v_dim + lane] = state28.y;
      out_value = fma(state28.y, fma(shared_q[113u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 114u * head_v_dim + lane] = state28.z;
      out_value = fma(state28.z, fma(shared_q[114u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 115u * head_v_dim + lane] = state28.w;
      out_value = fma(state28.w, fma(shared_q[115u], q_norm_scale, 0.0), out_value);
      state29 = fma(vec4<f32>(multiply_ordered(shared_k[116u], k_norm_scale), multiply_ordered(shared_k[117u], k_norm_scale), multiply_ordered(shared_k[118u], k_norm_scale), multiply_ordered(shared_k[119u], k_norm_scale)), vec4<f32>(delta), state29);
      recurrent_state[recurrent_head_base + 116u * head_v_dim + lane] = state29.x;
      out_value = fma(state29.x, fma(shared_q[116u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 117u * head_v_dim + lane] = state29.y;
      out_value = fma(state29.y, fma(shared_q[117u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 118u * head_v_dim + lane] = state29.z;
      out_value = fma(state29.z, fma(shared_q[118u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 119u * head_v_dim + lane] = state29.w;
      out_value = fma(state29.w, fma(shared_q[119u], q_norm_scale, 0.0), out_value);
      state30 = fma(vec4<f32>(multiply_ordered(shared_k[120u], k_norm_scale), multiply_ordered(shared_k[121u], k_norm_scale), multiply_ordered(shared_k[122u], k_norm_scale), multiply_ordered(shared_k[123u], k_norm_scale)), vec4<f32>(delta), state30);
      recurrent_state[recurrent_head_base + 120u * head_v_dim + lane] = state30.x;
      out_value = fma(state30.x, fma(shared_q[120u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 121u * head_v_dim + lane] = state30.y;
      out_value = fma(state30.y, fma(shared_q[121u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 122u * head_v_dim + lane] = state30.z;
      out_value = fma(state30.z, fma(shared_q[122u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 123u * head_v_dim + lane] = state30.w;
      out_value = fma(state30.w, fma(shared_q[123u], q_norm_scale, 0.0), out_value);
      state31 = fma(vec4<f32>(multiply_ordered(shared_k[124u], k_norm_scale), multiply_ordered(shared_k[125u], k_norm_scale), multiply_ordered(shared_k[126u], k_norm_scale), multiply_ordered(shared_k[127u], k_norm_scale)), vec4<f32>(delta), state31);
      recurrent_state[recurrent_head_base + 124u * head_v_dim + lane] = state31.x;
      out_value = fma(state31.x, fma(shared_q[124u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 125u * head_v_dim + lane] = state31.y;
      out_value = fma(state31.y, fma(shared_q[125u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 126u * head_v_dim + lane] = state31.z;
      out_value = fma(state31.z, fma(shared_q[126u], q_norm_scale, 0.0), out_value);
      recurrent_state[recurrent_head_base + 127u * head_v_dim + lane] = state31.w;
      out_value = fma(state31.w, fma(shared_q[127u], q_norm_scale, 0.0), out_value);
  }

  let out_sum_sq = reduce_sum(select(0.0, out_value * out_value, is_v_active), lane);
  let inv_rms = inverse_root_refined(out_sum_sq / f32(head_v_dim) + params.rms_norm_eps);

  if (is_v_active) {
    let z_row_base = head * head_v_dim;
    let z_packed_base = params.conv_dim + z_row_base;
    let z_index = select(z_row_base + lane, z_packed_base + lane, (params.packed_flags & 2u) != 0u);
    let gate = silu(f32(z_proj[z_index]));
    let norm_index = select(lane, head * head_v_dim + lane, params.norm_mode == 1u);
    var value = out_value;
    let factors = array<f32, 3>(inv_rms, norm_weight[norm_index], gate);
    for (var i = 0u; i < 3u; i = i + 1u) {
      value = fma(value, factors[i], 0.0);
    }
    output[out_row_base + lane] = value;
  }
}
