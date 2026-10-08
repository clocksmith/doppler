enable f16;

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
@group(0) @binding(1) var<storage, read> conv_out: array<f32>;
@group(0) @binding(2) var<storage, read> z_proj: array<f16>;
@group(0) @binding(3) var<storage, read> a_proj: array<f16>;
@group(0) @binding(4) var<storage, read> b_proj: array<f16>;
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
  return log(1.0 + exp_refined(x));
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
  let q_rep = max(params.q_rep, 1u);
  let src_head = head / q_rep;
  let q_base = src_head * head_k_dim;
  let k_base = params.q_size + src_head * head_k_dim;
  let v_base = params.q_size + params.k_size + head * head_v_dim;

  for (var token_idx: u32 = 0u; token_idx < params.num_tokens; token_idx = token_idx + 1u) {
    let conv_row_base = token_idx * params.conv_dim;
    let z_row_base = token_idx * params.value_dim + head * head_v_dim;
    let z_packed_base = token_idx * (params.conv_dim + params.value_dim) + params.conv_dim + head * head_v_dim;
    let ab_row_base = token_idx * params.num_v_heads + head;
    let out_row_base = token_idx * params.value_dim + head * head_v_dim;

    var q_norm_sq = 0.0;
    for (var d: u32 = vd; d < head_k_dim; d = d + WORKGROUP_SIZE) {
      let q_val = conv_out[conv_row_base + q_base + d];
      q_norm_sq = q_norm_sq + q_val * q_val;
    }
    shared_sq[vd] = q_norm_sq;
    workgroupBarrier();
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
      if (vd < stride) {
        shared_sq[vd] = shared_sq[vd] + shared_sq[vd + stride];
      }
      workgroupBarrier();
    }
    let q_norm_scale = head_scale * inverse_root_refined(shared_sq[0] + params.qk_l2norm_eps);
    // All invocations must consume the reduction before shared_sq is reused.
    workgroupBarrier();

    var k_norm_sq = 0.0;
    for (var d: u32 = vd; d < head_k_dim; d = d + WORKGROUP_SIZE) {
      let k_val = conv_out[conv_row_base + k_base + d];
      k_norm_sq = k_norm_sq + k_val * k_val;
    }
    shared_sq[vd] = k_norm_sq;
    workgroupBarrier();
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
      if (vd < stride) {
        shared_sq[vd] = shared_sq[vd] + shared_sq[vd + stride];
      }
      workgroupBarrier();
    }
    let k_norm_scale = inverse_root_refined(shared_sq[0] + params.qk_l2norm_eps);
    // All invocations must consume the reduction before shared_sq is reused.
    workgroupBarrier();
    let b_index = select(ab_row_base, params.b_proj_offset_elements + ab_row_base, (params.packed_flags & 1u) != 0u);
    let beta = sigmoid_refined(f32(b_proj[b_index]));
    let g = -exp_refined(a_log[head]) * softplus(f32(a_proj[ab_row_base]) + dt_bias[head]);
    let g_exp = exp_refined(g);

    if (is_active) {
      for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
        let state_idx = recurrent_head_base + kd * head_v_dim + vd;
        recurrent_state[state_idx] = recurrent_state[state_idx] * g_exp;
      }
    }
    var kv_mem = 0.0;
    if (is_active) {
      for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
        let k_normed = conv_out[conv_row_base + k_base + kd] * k_norm_scale;
        let state_idx = recurrent_head_base + kd * head_v_dim + vd;
        kv_mem = kv_mem + recurrent_state[state_idx] * k_normed;
      }
      let delta = (conv_out[conv_row_base + v_base + vd] - kv_mem) * beta;
      for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
        let k_normed = conv_out[conv_row_base + k_base + kd] * k_norm_scale;
        let state_idx = recurrent_head_base + kd * head_v_dim + vd;
        recurrent_state[state_idx] = recurrent_state[state_idx] + k_normed * delta;
      }
    }

    var out_value = 0.0;
    if (is_active) {
      for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
        let q_normed = conv_out[conv_row_base + q_base + kd] * q_norm_scale;
        let state_idx = recurrent_head_base + kd * head_v_dim + vd;
        out_value = out_value + recurrent_state[state_idx] * q_normed;
      }
    }
    if (is_active) {
      output[out_row_base + vd] = out_value;
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
    // All invocations must consume the reduction before shared_sq is reused.
    workgroupBarrier();

    if (is_active) {
      let z_index = select(z_row_base + vd, z_packed_base + vd, (params.packed_flags & 2u) != 0u);
      let gate = silu(f32(z_proj[z_index]));
      let norm_index = select(vd, head * head_v_dim + vd, params.norm_mode == 1u);
      output[out_row_base + vd] = (output[out_row_base + vd] * inv_rms) * norm_weight[norm_index] * gate;
    }
  }
}
