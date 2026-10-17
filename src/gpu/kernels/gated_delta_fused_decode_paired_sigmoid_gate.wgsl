// AUTO-GENERATED from src/gpu/kernels/gated_delta_fused_decode.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// Gated RMSNorm with a sigmoid gate: norm(x) * w * sigmoid(z). Silu lanes use the source.
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

// BEGIN GENERATED COMPENSATED STATE
// Compensated recurrent-state arithmetic; finite operands and intermediate sums/products.
// Overflow propagates nonfinite values; this is not an extended-exponent format.
// This is a generator input, embedded verbatim in the four gated-delta kernels.
fn add_u64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low = a.x + b.x;
    return vec2<u32>(low, a.y + b.y + u32(low < a.x));
}
fn negate_u64(a: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(0u - a.x, ~a.y + u32(a.x == 0u));
}
fn aligned_significand(value: f32, exponent: u32) -> vec2<u32> {
    let bits = bitcast<u32>(value);
    let stored_exponent = (bits >> 23u) & 255u;
    let mantissa = (bits & 0x007fffffu) | select(0u, 0x00800000u, stored_exponent != 0u);
    let shift = exponent - max(stored_exponent, 1u);
    var magnitude = vec2<u32>(0u);
    if (shift == 0u) { magnitude.y = mantissa; }
    else if (shift < 32u) { magnitude = vec2<u32>(mantissa << (32u - shift), mantissa >> shift); }
    else if (shift < 64u) { magnitude.x = mantissa >> (shift - 32u); }
    return select(magnitude, negate_u64(magnitude), (bits & 0x80000000u) != 0u);
}
fn addition_residual(a: f32, b: f32, sum: f32) -> f32 {
    if (bitcast<u32>(sum) == bitcast<u32>(a)) { return b; }
    if (bitcast<u32>(sum) == bitcast<u32>(b)) { return a; }
    let exponent = max(max((bitcast<u32>(a) >> 23u) & 255u, (bitcast<u32>(b) >> 23u) & 255u),
                       max((bitcast<u32>(sum) >> 23u) & 255u, 1u));
    if (exponent == 255u) { return 0.0; }
    let residual = add_u64(add_u64(aligned_significand(a, exponent), aligned_significand(b, exponent)),
                            negate_u64(aligned_significand(sum, exponent)));
    let negative = (residual.y & 0x80000000u) != 0u;
    let magnitude = select(residual, negate_u64(residual), negative);
    return scaled_significand(magnitude, i32(exponent) - 182, negative);
}


// Construct the binary32 result, including nearest-even subnormal rounding,
// without an arithmetic operation that may flush subnormal operands/results.
fn rounded_shift(value: vec2<u32>, shift: u32) -> u32 {
    if (shift == 0u) { return value.x; }
    if (shift >= 64u) { return 0u; }
    var quotient: u32;
    var above: bool;
    var tie: bool;
    if (shift < 32u) {
        quotient = (value.x >> shift) | (value.y << (32u - shift));
        let remainder = value.x & ((1u << shift) - 1u);
        let halfway = 1u << (shift - 1u);
        above = remainder > halfway; tie = remainder == halfway;
    } else if (shift == 32u) {
        quotient = value.y; above = value.x > 0x80000000u; tie = value.x == 0x80000000u;
    } else {
        let high_shift = shift - 32u;
        quotient = value.y >> high_shift;
        let remainder = value.y & ((1u << high_shift) - 1u);
        let halfway = 1u << (high_shift - 1u);
        above = remainder > halfway || (remainder == halfway && value.x != 0u);
        tie = remainder == halfway && value.x == 0u;
    }
    return quotient + u32(above || (tie && (quotient & 1u) != 0u));
}
fn scaled_significand(magnitude: vec2<u32>, exponent: i32, negative: bool) -> f32 {
    let sign = select(0u, 0x80000000u, negative);
    if (all(magnitude == vec2<u32>(0u))) { return bitcast<f32>(sign); }
    let top = select(firstLeadingBit(magnitude.x), firstLeadingBit(magnitude.y) + 32u, magnitude.y != 0u);
    var unbiased = exponent + i32(top);
    var bits: u32;
    if (unbiased >= -126) {
        var significand = select(magnitude.x << (23u - min(top, 23u)), rounded_shift(magnitude, top - min(top, 23u)), top > 23u);
        if (significand >= 0x01000000u) { significand >>= 1u; unbiased += 1; }
        bits = (u32(unbiased + 127) << 23u) | (significand & 0x007fffffu);
    } else {
        let shift = exponent + 149;
        if (shift >= 0) { bits = magnitude.x << u32(shift); }
        else { bits = rounded_shift(magnitude, u32(-shift)); }
    }
    return bitcast<f32>(sign | bits);
}

// Exact product residual from the 24-bit significands, avoiding a floating
// multiply/subtract identity that a relaxed backend may reassociate.
fn product_residual(a: f32, b: f32, product: f32) -> f32 {
    let ab = bitcast<u32>(a); let bb = bitcast<u32>(b); let pb = bitcast<u32>(product);
    let ae = (ab >> 23u) & 255u; let be = (bb >> 23u) & 255u; let pe = (pb >> 23u) & 255u;
    if (ae == 255u || be == 255u || pe == 255u || (ab & 0x7fffffffu) == 0u || (bb & 0x7fffffffu) == 0u) { return 0.0; }
    let am = (ab & 0x007fffffu) | select(0u, 0x00800000u, ae != 0u);
    let bm = (bb & 0x007fffffu) | select(0u, 0x00800000u, be != 0u);
    let pm = (pb & 0x007fffffu) | select(0u, 0x00800000u, pe != 0u);
    let al = am & 65535u; let ah = am >> 16u;
    let bl = bm & 65535u; let bh = bm >> 16u;
    let cross = al * bh + ah * bl;
    let exact = add_u64(vec2<u32>(al * bl, ah * bh), vec2<u32>(cross << 16u, cross >> 16u));
    if ((pb & 0x7fffffffu) == 0u) {
        return scaled_significand(exact, i32(max(ae, 1u)) + i32(max(be, 1u)) - 300, ((ab ^ bb) & 0x80000000u) != 0u);
    }
    let shift = i32(max(pe, 1u)) + 150 - i32(max(ae, 1u)) - i32(max(be, 1u));
    if (shift >= 64) { return bitcast<f32>(pb ^ 0x80000000u); }
    if (shift < 0) { return 0.0; }
    var rounded = vec2<u32>(0u);
    if (shift == 0) { rounded.x = pm; }
    else if (shift < 32) { rounded = vec2<u32>(pm << u32(shift), pm >> (32u - u32(shift))); }
    else { rounded.y = pm << (u32(shift) - 32u); }
    let residual = add_u64(exact, negate_u64(rounded));
    let negative = (residual.y & 0x80000000u) != 0u;
    let magnitude = select(residual, negate_u64(residual), negative);
    let sign = negative != (((ab ^ bb) & 0x80000000u) != 0u);
    return scaled_significand(magnitude, i32(max(ae, 1u)) + i32(max(be, 1u)) - 300, sign);
}


// Normalized two-component arithmetic. Integer residual recovery prevents
// relaxed floating-point reassociation from deleting a correction term.
fn add_preserving_underflow(a: f32, b: f32) -> f32 {
    let rounded = a + b;
    if ((bitcast<u32>(rounded) & 0x7fffffffu) == 0u) {
        return addition_residual(a, b, rounded);
    }
    return rounded;
}
fn pair_normalize(high: f32, low: f32) -> vec2<f32> {
    let rounded = high + low;
    let residual = addition_residual(high, low, rounded);
    if ((bitcast<u32>(rounded) & 0x7fffffffu) == 0u) {
        return vec2<f32>(residual, 0.0);
    }
    return vec2<f32>(rounded, residual);
}
fn pair_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let high = a.x + b.x;
    let error = addition_residual(a.x, b.x, high);
    let low = add_preserving_underflow(a.y, b.y);
    let joined = add_preserving_underflow(error, low);
    let tail = add_preserving_underflow(addition_residual(a.y, b.y, low), addition_residual(error, low, joined));
    let result = pair_normalize(high, joined);
    return pair_normalize(result.x, add_preserving_underflow(result.y, tail));
}
fn pair_mul_scalar(a: vec2<f32>, b: f32) -> vec2<f32> {
    let high = a.x * b;
    let low = a.y * b;
    return pair_add(vec2<f32>(high, product_residual(a.x, b, high)),
                    vec2<f32>(low, product_residual(a.y, b, low)));
}
// END GENERATED COMPENSATED STATE

fn state_load(index: u32) -> vec2<f32> {
  let base = index * RECURRENT_STATE_COMPONENTS;
  if (RECURRENT_STATE_COMPONENTS == 2u) {
    return vec2<f32>(recurrent_state[base], recurrent_state[base + 1u]);
  }
  return vec2<f32>(recurrent_state[base], 0.0);
}
fn state_store(index: u32, value: vec2<f32>) {
  let base = index * RECURRENT_STATE_COMPONENTS;
  recurrent_state[base] = value.x;
  if (RECURRENT_STATE_COMPONENTS == 2u) {
    recurrent_state[base + 1u] = value.y;
  }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wid.x;
  let lane = lid.x;
  if (head >= params.num_v_heads || params.num_tokens != 1u || params.q_rep != 1u) {
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

  if (is_v_active) {
    for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
      let state_idx = recurrent_head_base + kd * head_v_dim + lane;
      if (RECURRENT_STATE_COMPONENTS == 2u) {
          state_store(state_idx, pair_mul_scalar(state_load(state_idx), g_exp));
        } else {
          recurrent_state[state_idx] = recurrent_state[state_idx] * g_exp;
        }
    }
  }

  var kv_mem = 0.0;
    var kv_pair = vec2<f32>(0.0);
  if (is_v_active) {
    for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
      let k_normed = multiply_ordered(shared_k[kd], k_norm_scale);
      let state_idx = recurrent_head_base + kd * head_v_dim + lane;
      if (RECURRENT_STATE_COMPONENTS == 2u) {
          kv_pair = pair_add(kv_pair, pair_mul_scalar(state_load(state_idx), k_normed));
        } else {
          kv_mem = fma(recurrent_state[state_idx], k_normed, kv_mem);
        }
    }
    let delta = multiply_ordered(v_val - kv_mem, beta);
      var delta_pair = vec2<f32>(0.0);
      if (RECURRENT_STATE_COMPONENTS == 2u) {
        delta_pair = pair_mul_scalar(pair_add(vec2<f32>(v_val, 0.0), -kv_pair), beta);
      }
    for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
      let k_normed = multiply_ordered(shared_k[kd], k_norm_scale);
      let state_idx = recurrent_head_base + kd * head_v_dim + lane;
      if (RECURRENT_STATE_COMPONENTS == 2u) {
          state_store(state_idx, pair_add(state_load(state_idx), pair_mul_scalar(delta_pair, k_normed)));
        } else {
          recurrent_state[state_idx] = fma(k_normed, delta, recurrent_state[state_idx]);
        }
    }
  }

  var out_value = 0.0;
    var out_pair = vec2<f32>(0.0);
  if (is_v_active) {
    for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
      let q_normed = fma(shared_q[kd], q_norm_scale, 0.0);
      let state_idx = recurrent_head_base + kd * head_v_dim + lane;
      if (RECURRENT_STATE_COMPONENTS == 2u) {
          out_pair = pair_add(out_pair, pair_mul_scalar(state_load(state_idx), q_normed));
        } else {
          out_value = fma(recurrent_state[state_idx], q_normed, out_value);
        }
    }
  }

  if (RECURRENT_STATE_COMPONENTS == 2u) {
    out_value = out_pair.x + out_pair.y;
  }
  let out_sum_sq = reduce_sum(select(0.0, out_value * out_value, is_v_active), lane);
  let inv_rms = inverse_root_refined(out_sum_sq / f32(head_v_dim) + params.rms_norm_eps);

  if (is_v_active) {
    let z_row_base = head * head_v_dim;
    let z_packed_base = params.conv_dim + z_row_base;
    let z_index = select(z_row_base + lane, z_packed_base + lane, (params.packed_flags & 2u) != 0u);
    let gate = sigmoid_refined(f32(z_proj[z_index]));
    let norm_index = select(lane, head * head_v_dim + lane, params.norm_mode == 1u);
    var value = out_value;
    let factors = array<f32, 3>(inv_rms, norm_weight[norm_index], gate);
    for (var i = 0u; i < 3u; i = i + 1u) {
      value = fma(value, factors[i], 0.0);
    }
    output[out_row_base + lane] = value;
  }
}
