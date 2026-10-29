// AUTO-GENERATED from src/gpu/kernels/gated_delta_recurrent.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// F32 fixed-width prefill with explicit vector register slots.
override RECURRENT_STATE_COMPONENTS: u32 = 1u;
const HEAD_K_DIM: u32 = 128u;

override WORKGROUP_SIZE: u32 = 32u;

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


var<workgroup> shared_coefficients: vec4<f32>;
var<workgroup> q_squared: array<f32, HEAD_K_DIM>;
var<workgroup> k_squared: array<f32, HEAD_K_DIM>;
var<workgroup> shared_k_sq: array<f32, WORKGROUP_SIZE>;
var<workgroup> normalized_q: array<vec4<f32>, HEAD_K_DIM / 4u>;
var<workgroup> normalized_k: array<vec4<f32>, HEAD_K_DIM / 4u>;
struct RetainedState {
  s0: vec4<f32>,
  s1: vec4<f32>,
  s2: vec4<f32>,
  s3: vec4<f32>,
  s4: vec4<f32>,
  s5: vec4<f32>,
  s6: vec4<f32>,
  s7: vec4<f32>,
  s8: vec4<f32>,
  s9: vec4<f32>,
  s10: vec4<f32>,
  s11: vec4<f32>,
  s12: vec4<f32>,
  s13: vec4<f32>,
  s14: vec4<f32>,
  s15: vec4<f32>,
  s16: vec4<f32>,
  s17: vec4<f32>,
  s18: vec4<f32>,
  s19: vec4<f32>,
  s20: vec4<f32>,
  s21: vec4<f32>,
  s22: vec4<f32>,
  s23: vec4<f32>,
  s24: vec4<f32>,
  s25: vec4<f32>,
  s26: vec4<f32>,
  s27: vec4<f32>,
  s28: vec4<f32>,
  s29: vec4<f32>,
  s30: vec4<f32>,
  s31: vec4<f32>,
}
fn load_state4(base: u32, stride: u32) -> vec4<f32> {
  return vec4<f32>(recurrent_state[base], recurrent_state[base + stride],
    recurrent_state[base + 2u * stride], recurrent_state[base + 3u * stride]);
}
fn store_state4(base: u32, stride: u32, value: vec4<f32>) {
  recurrent_state[base] = value.x;
  recurrent_state[base + stride] = value.y;
  recurrent_state[base + 2u * stride] = value.z;
  recurrent_state[base + 3u * stride] = value.w;
}
fn accumulate_four(state: vec4<f32>, input: vec4<f32>, initial: f32) -> f32 {
  var sum = fma(state.x, input.x, initial);
  sum = fma(state.y, input.y, sum);
  sum = fma(state.z, input.z, sum);
  return fma(state.w, input.w, sum);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wid.x;
  let lane = lid.x;
  let vd = wid.y * WORKGROUP_SIZE + lane;
  if (head >= params.num_v_heads) {
    return;
  }

  let head_k_dim = HEAD_K_DIM;
  let head_v_dim = params.head_v_dim;
  let is_active = vd < head_v_dim;
  let head_scale = inverseSqrt(f32(head_k_dim));
  let recurrent_head_base = head * head_k_dim * head_v_dim;
  let q_rep = max(params.q_rep, 1u);
  let src_head = head / q_rep;
  let q_base = src_head * head_k_dim;
  let k_base = params.q_size + src_head * head_k_dim;
  let v_base = params.q_size + params.k_size + head * head_v_dim;

  var state: RetainedState;
  if (is_active) {
    state.s0 = load_state4(recurrent_head_base + 0u * head_v_dim + vd, head_v_dim);
    state.s1 = load_state4(recurrent_head_base + 4u * head_v_dim + vd, head_v_dim);
    state.s2 = load_state4(recurrent_head_base + 8u * head_v_dim + vd, head_v_dim);
    state.s3 = load_state4(recurrent_head_base + 12u * head_v_dim + vd, head_v_dim);
    state.s4 = load_state4(recurrent_head_base + 16u * head_v_dim + vd, head_v_dim);
    state.s5 = load_state4(recurrent_head_base + 20u * head_v_dim + vd, head_v_dim);
    state.s6 = load_state4(recurrent_head_base + 24u * head_v_dim + vd, head_v_dim);
    state.s7 = load_state4(recurrent_head_base + 28u * head_v_dim + vd, head_v_dim);
    state.s8 = load_state4(recurrent_head_base + 32u * head_v_dim + vd, head_v_dim);
    state.s9 = load_state4(recurrent_head_base + 36u * head_v_dim + vd, head_v_dim);
    state.s10 = load_state4(recurrent_head_base + 40u * head_v_dim + vd, head_v_dim);
    state.s11 = load_state4(recurrent_head_base + 44u * head_v_dim + vd, head_v_dim);
    state.s12 = load_state4(recurrent_head_base + 48u * head_v_dim + vd, head_v_dim);
    state.s13 = load_state4(recurrent_head_base + 52u * head_v_dim + vd, head_v_dim);
    state.s14 = load_state4(recurrent_head_base + 56u * head_v_dim + vd, head_v_dim);
    state.s15 = load_state4(recurrent_head_base + 60u * head_v_dim + vd, head_v_dim);
    state.s16 = load_state4(recurrent_head_base + 64u * head_v_dim + vd, head_v_dim);
    state.s17 = load_state4(recurrent_head_base + 68u * head_v_dim + vd, head_v_dim);
    state.s18 = load_state4(recurrent_head_base + 72u * head_v_dim + vd, head_v_dim);
    state.s19 = load_state4(recurrent_head_base + 76u * head_v_dim + vd, head_v_dim);
    state.s20 = load_state4(recurrent_head_base + 80u * head_v_dim + vd, head_v_dim);
    state.s21 = load_state4(recurrent_head_base + 84u * head_v_dim + vd, head_v_dim);
    state.s22 = load_state4(recurrent_head_base + 88u * head_v_dim + vd, head_v_dim);
    state.s23 = load_state4(recurrent_head_base + 92u * head_v_dim + vd, head_v_dim);
    state.s24 = load_state4(recurrent_head_base + 96u * head_v_dim + vd, head_v_dim);
    state.s25 = load_state4(recurrent_head_base + 100u * head_v_dim + vd, head_v_dim);
    state.s26 = load_state4(recurrent_head_base + 104u * head_v_dim + vd, head_v_dim);
    state.s27 = load_state4(recurrent_head_base + 108u * head_v_dim + vd, head_v_dim);
    state.s28 = load_state4(recurrent_head_base + 112u * head_v_dim + vd, head_v_dim);
    state.s29 = load_state4(recurrent_head_base + 116u * head_v_dim + vd, head_v_dim);
    state.s30 = load_state4(recurrent_head_base + 120u * head_v_dim + vd, head_v_dim);
    state.s31 = load_state4(recurrent_head_base + 124u * head_v_dim + vd, head_v_dim);
  }

  for (var token_idx: u32 = 0u; token_idx < params.num_tokens; token_idx = token_idx + 1u) {
    let conv_row_base = token_idx * params.conv_dim;
    let z_row_base = token_idx * params.value_dim + head * head_v_dim;
    let z_packed_base = token_idx * (params.conv_dim + params.value_dim) + params.conv_dim + head * head_v_dim;
    let ab_row_base = token_idx * params.num_v_heads + head;
    let out_row_base = token_idx * params.value_dim + head * head_v_dim;

    for (var d = lane; d < HEAD_K_DIM; d += WORKGROUP_SIZE) {
      let q = conv_out[conv_row_base + q_base + d];
      let k = conv_out[conv_row_base + k_base + d];
      q_squared[d] = q * q;
      k_squared[d] = k * k;
    }
    workgroupBarrier();
    let q_norm_sq = (q_squared[lane] + q_squared[lane + 64u])
      + (q_squared[lane + 32u] + q_squared[lane + 96u]);
    let k_norm_sq = (k_squared[lane] + k_squared[lane + 64u])
      + (k_squared[lane + 32u] + k_squared[lane + 96u]);
    shared_sq[lane] = q_norm_sq;
    shared_k_sq[lane] = k_norm_sq;
    workgroupBarrier();
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) {
        shared_sq[lane] = shared_sq[lane] + shared_sq[lane + stride];
        shared_k_sq[lane] = shared_k_sq[lane] + shared_k_sq[lane + stride];
      }
      workgroupBarrier();
    }
    if (lane == 0u) {
      shared_coefficients.x = head_scale * inverse_root_refined(shared_sq[0] + params.qk_l2norm_eps);
      shared_coefficients.y = inverse_root_refined(shared_k_sq[0] + params.qk_l2norm_eps);
      let b_index = select(ab_row_base, params.b_proj_offset_elements + ab_row_base, (params.packed_flags & 1u) != 0u);
      shared_coefficients.z = sigmoid_refined(f32(b_proj[b_index]));
      let g = -exp_refined(a_log[head]) * softplus(f32(a_proj[ab_row_base]) + dt_bias[head]);
      shared_coefficients.w = exp_refined(g);
    }
    workgroupBarrier();
    let q_norm_scale = shared_coefficients.x;
    let k_norm_scale = shared_coefficients.y;
    if (lane < HEAD_K_DIM / 4u) {
      let q_base4 = conv_row_base + q_base + lane * 4u;
      let k_base4 = conv_row_base + k_base + lane * 4u;
      normalized_q[lane] = fma(vec4<f32>(conv_out[q_base4], conv_out[q_base4 + 1u],
        conv_out[q_base4 + 2u], conv_out[q_base4 + 3u]), vec4<f32>(q_norm_scale), vec4<f32>(0.0));
      normalized_k[lane] = vec4<f32>(multiply_ordered(conv_out[k_base4], k_norm_scale),
        multiply_ordered(conv_out[k_base4 + 1u], k_norm_scale),
        multiply_ordered(conv_out[k_base4 + 2u], k_norm_scale),
        multiply_ordered(conv_out[k_base4 + 3u], k_norm_scale));
    }
    workgroupBarrier();
    let beta = shared_coefficients.z;
    let g_exp = shared_coefficients.w;

    var kv_mem = 0.0;
    var out_value = 0.0;
    if (is_active) {
      state.s0 = state.s0 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s0, normalized_k[0], kv_mem);
      state.s1 = state.s1 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s1, normalized_k[1], kv_mem);
      state.s2 = state.s2 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s2, normalized_k[2], kv_mem);
      state.s3 = state.s3 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s3, normalized_k[3], kv_mem);
      state.s4 = state.s4 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s4, normalized_k[4], kv_mem);
      state.s5 = state.s5 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s5, normalized_k[5], kv_mem);
      state.s6 = state.s6 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s6, normalized_k[6], kv_mem);
      state.s7 = state.s7 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s7, normalized_k[7], kv_mem);
      state.s8 = state.s8 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s8, normalized_k[8], kv_mem);
      state.s9 = state.s9 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s9, normalized_k[9], kv_mem);
      state.s10 = state.s10 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s10, normalized_k[10], kv_mem);
      state.s11 = state.s11 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s11, normalized_k[11], kv_mem);
      state.s12 = state.s12 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s12, normalized_k[12], kv_mem);
      state.s13 = state.s13 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s13, normalized_k[13], kv_mem);
      state.s14 = state.s14 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s14, normalized_k[14], kv_mem);
      state.s15 = state.s15 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s15, normalized_k[15], kv_mem);
      state.s16 = state.s16 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s16, normalized_k[16], kv_mem);
      state.s17 = state.s17 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s17, normalized_k[17], kv_mem);
      state.s18 = state.s18 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s18, normalized_k[18], kv_mem);
      state.s19 = state.s19 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s19, normalized_k[19], kv_mem);
      state.s20 = state.s20 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s20, normalized_k[20], kv_mem);
      state.s21 = state.s21 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s21, normalized_k[21], kv_mem);
      state.s22 = state.s22 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s22, normalized_k[22], kv_mem);
      state.s23 = state.s23 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s23, normalized_k[23], kv_mem);
      state.s24 = state.s24 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s24, normalized_k[24], kv_mem);
      state.s25 = state.s25 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s25, normalized_k[25], kv_mem);
      state.s26 = state.s26 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s26, normalized_k[26], kv_mem);
      state.s27 = state.s27 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s27, normalized_k[27], kv_mem);
      state.s28 = state.s28 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s28, normalized_k[28], kv_mem);
      state.s29 = state.s29 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s29, normalized_k[29], kv_mem);
      state.s30 = state.s30 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s30, normalized_k[30], kv_mem);
      state.s31 = state.s31 * vec4<f32>(g_exp);
      kv_mem = accumulate_four(state.s31, normalized_k[31], kv_mem);
      let delta = multiply_ordered(conv_out[conv_row_base + v_base + vd] - kv_mem, beta);
      state.s0 = fma(normalized_k[0], vec4<f32>(delta), state.s0);
      out_value = accumulate_four(state.s0, normalized_q[0], out_value);
      state.s1 = fma(normalized_k[1], vec4<f32>(delta), state.s1);
      out_value = accumulate_four(state.s1, normalized_q[1], out_value);
      state.s2 = fma(normalized_k[2], vec4<f32>(delta), state.s2);
      out_value = accumulate_four(state.s2, normalized_q[2], out_value);
      state.s3 = fma(normalized_k[3], vec4<f32>(delta), state.s3);
      out_value = accumulate_four(state.s3, normalized_q[3], out_value);
      state.s4 = fma(normalized_k[4], vec4<f32>(delta), state.s4);
      out_value = accumulate_four(state.s4, normalized_q[4], out_value);
      state.s5 = fma(normalized_k[5], vec4<f32>(delta), state.s5);
      out_value = accumulate_four(state.s5, normalized_q[5], out_value);
      state.s6 = fma(normalized_k[6], vec4<f32>(delta), state.s6);
      out_value = accumulate_four(state.s6, normalized_q[6], out_value);
      state.s7 = fma(normalized_k[7], vec4<f32>(delta), state.s7);
      out_value = accumulate_four(state.s7, normalized_q[7], out_value);
      state.s8 = fma(normalized_k[8], vec4<f32>(delta), state.s8);
      out_value = accumulate_four(state.s8, normalized_q[8], out_value);
      state.s9 = fma(normalized_k[9], vec4<f32>(delta), state.s9);
      out_value = accumulate_four(state.s9, normalized_q[9], out_value);
      state.s10 = fma(normalized_k[10], vec4<f32>(delta), state.s10);
      out_value = accumulate_four(state.s10, normalized_q[10], out_value);
      state.s11 = fma(normalized_k[11], vec4<f32>(delta), state.s11);
      out_value = accumulate_four(state.s11, normalized_q[11], out_value);
      state.s12 = fma(normalized_k[12], vec4<f32>(delta), state.s12);
      out_value = accumulate_four(state.s12, normalized_q[12], out_value);
      state.s13 = fma(normalized_k[13], vec4<f32>(delta), state.s13);
      out_value = accumulate_four(state.s13, normalized_q[13], out_value);
      state.s14 = fma(normalized_k[14], vec4<f32>(delta), state.s14);
      out_value = accumulate_four(state.s14, normalized_q[14], out_value);
      state.s15 = fma(normalized_k[15], vec4<f32>(delta), state.s15);
      out_value = accumulate_four(state.s15, normalized_q[15], out_value);
      state.s16 = fma(normalized_k[16], vec4<f32>(delta), state.s16);
      out_value = accumulate_four(state.s16, normalized_q[16], out_value);
      state.s17 = fma(normalized_k[17], vec4<f32>(delta), state.s17);
      out_value = accumulate_four(state.s17, normalized_q[17], out_value);
      state.s18 = fma(normalized_k[18], vec4<f32>(delta), state.s18);
      out_value = accumulate_four(state.s18, normalized_q[18], out_value);
      state.s19 = fma(normalized_k[19], vec4<f32>(delta), state.s19);
      out_value = accumulate_four(state.s19, normalized_q[19], out_value);
      state.s20 = fma(normalized_k[20], vec4<f32>(delta), state.s20);
      out_value = accumulate_four(state.s20, normalized_q[20], out_value);
      state.s21 = fma(normalized_k[21], vec4<f32>(delta), state.s21);
      out_value = accumulate_four(state.s21, normalized_q[21], out_value);
      state.s22 = fma(normalized_k[22], vec4<f32>(delta), state.s22);
      out_value = accumulate_four(state.s22, normalized_q[22], out_value);
      state.s23 = fma(normalized_k[23], vec4<f32>(delta), state.s23);
      out_value = accumulate_four(state.s23, normalized_q[23], out_value);
      state.s24 = fma(normalized_k[24], vec4<f32>(delta), state.s24);
      out_value = accumulate_four(state.s24, normalized_q[24], out_value);
      state.s25 = fma(normalized_k[25], vec4<f32>(delta), state.s25);
      out_value = accumulate_four(state.s25, normalized_q[25], out_value);
      state.s26 = fma(normalized_k[26], vec4<f32>(delta), state.s26);
      out_value = accumulate_four(state.s26, normalized_q[26], out_value);
      state.s27 = fma(normalized_k[27], vec4<f32>(delta), state.s27);
      out_value = accumulate_four(state.s27, normalized_q[27], out_value);
      state.s28 = fma(normalized_k[28], vec4<f32>(delta), state.s28);
      out_value = accumulate_four(state.s28, normalized_q[28], out_value);
      state.s29 = fma(normalized_k[29], vec4<f32>(delta), state.s29);
      out_value = accumulate_four(state.s29, normalized_q[29], out_value);
      state.s30 = fma(normalized_k[30], vec4<f32>(delta), state.s30);
      out_value = accumulate_four(state.s30, normalized_q[30], out_value);
      state.s31 = fma(normalized_k[31], vec4<f32>(delta), state.s31);
      out_value = accumulate_four(state.s31, normalized_q[31], out_value);
    }
    if (is_active) {
      output[out_row_base + vd] = out_value;
    }

    // Every slice must finish reading normalized Q/K before the next token overwrites them.
    workgroupBarrier();
  }

  if (is_active) {
    store_state4(recurrent_head_base + 0u * head_v_dim + vd, head_v_dim, state.s0);
    store_state4(recurrent_head_base + 4u * head_v_dim + vd, head_v_dim, state.s1);
    store_state4(recurrent_head_base + 8u * head_v_dim + vd, head_v_dim, state.s2);
    store_state4(recurrent_head_base + 12u * head_v_dim + vd, head_v_dim, state.s3);
    store_state4(recurrent_head_base + 16u * head_v_dim + vd, head_v_dim, state.s4);
    store_state4(recurrent_head_base + 20u * head_v_dim + vd, head_v_dim, state.s5);
    store_state4(recurrent_head_base + 24u * head_v_dim + vd, head_v_dim, state.s6);
    store_state4(recurrent_head_base + 28u * head_v_dim + vd, head_v_dim, state.s7);
    store_state4(recurrent_head_base + 32u * head_v_dim + vd, head_v_dim, state.s8);
    store_state4(recurrent_head_base + 36u * head_v_dim + vd, head_v_dim, state.s9);
    store_state4(recurrent_head_base + 40u * head_v_dim + vd, head_v_dim, state.s10);
    store_state4(recurrent_head_base + 44u * head_v_dim + vd, head_v_dim, state.s11);
    store_state4(recurrent_head_base + 48u * head_v_dim + vd, head_v_dim, state.s12);
    store_state4(recurrent_head_base + 52u * head_v_dim + vd, head_v_dim, state.s13);
    store_state4(recurrent_head_base + 56u * head_v_dim + vd, head_v_dim, state.s14);
    store_state4(recurrent_head_base + 60u * head_v_dim + vd, head_v_dim, state.s15);
    store_state4(recurrent_head_base + 64u * head_v_dim + vd, head_v_dim, state.s16);
    store_state4(recurrent_head_base + 68u * head_v_dim + vd, head_v_dim, state.s17);
    store_state4(recurrent_head_base + 72u * head_v_dim + vd, head_v_dim, state.s18);
    store_state4(recurrent_head_base + 76u * head_v_dim + vd, head_v_dim, state.s19);
    store_state4(recurrent_head_base + 80u * head_v_dim + vd, head_v_dim, state.s20);
    store_state4(recurrent_head_base + 84u * head_v_dim + vd, head_v_dim, state.s21);
    store_state4(recurrent_head_base + 88u * head_v_dim + vd, head_v_dim, state.s22);
    store_state4(recurrent_head_base + 92u * head_v_dim + vd, head_v_dim, state.s23);
    store_state4(recurrent_head_base + 96u * head_v_dim + vd, head_v_dim, state.s24);
    store_state4(recurrent_head_base + 100u * head_v_dim + vd, head_v_dim, state.s25);
    store_state4(recurrent_head_base + 104u * head_v_dim + vd, head_v_dim, state.s26);
    store_state4(recurrent_head_base + 108u * head_v_dim + vd, head_v_dim, state.s27);
    store_state4(recurrent_head_base + 112u * head_v_dim + vd, head_v_dim, state.s28);
    store_state4(recurrent_head_base + 116u * head_v_dim + vd, head_v_dim, state.s29);
    store_state4(recurrent_head_base + 120u * head_v_dim + vd, head_v_dim, state.s30);
    store_state4(recurrent_head_base + 124u * head_v_dim + vd, head_v_dim, state.s31);
  }
}
