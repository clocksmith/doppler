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
@group(0) @binding(1) var<storage, read> qkv: array<f16>;
@group(0) @binding(2) var<storage, read> z_proj: array<f16>;
@group(0) @binding(3) var<storage, read> ab_proj: array<f16>;
@group(0) @binding(4) var<storage, read> conv_weight: array<f32>;
@group(0) @binding(5) var<storage, read_write> conv_state: array<f32>;
@group(0) @binding(6) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(7) var<storage, read> a_neg_exp: array<f32>;
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
    return exp(x);
  }
  return log(1.0 + exp(x));
}

fn silu(x: f32) -> f32 {
  if (x >= 0.0) {
    let z = exp(-x);
    return x / (1.0 + z);
  }
  let z = exp(x);
  return x * z / (1.0 + z);
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
  return shared_reduce[0];
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
  let q_norm_scale = head_scale / sqrt(q_norm_sq + params.qk_l2norm_eps);
  let k_norm_sq = reduce_sum(select(0.0, k_val * k_val, is_k_active), lane);
  let k_norm_scale = inverseSqrt(k_norm_sq + params.qk_l2norm_eps);

  let ab_row_base = head;
  let b_index = params.b_proj_offset_elements + ab_row_base;
  let beta = 1.0 / (1.0 + exp(-f32(ab_proj[b_index])));
  let g = a_neg_exp[head] * softplus(f32(ab_proj[ab_row_base]) + dt_bias[head]);
  let g_exp = exp(g);

  if (is_v_active) {
    for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
      let state_idx = recurrent_head_base + kd * head_v_dim + lane;
      recurrent_state[state_idx] = recurrent_state[state_idx] * g_exp;
    }
  }

  var kv_mem = 0.0;
  if (is_v_active) {
    for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
      let k_normed = shared_k[kd] * k_norm_scale;
      let state_idx = recurrent_head_base + kd * head_v_dim + lane;
      kv_mem = kv_mem + recurrent_state[state_idx] * k_normed;
    }
    let delta = (v_val - kv_mem) * beta;
    for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
      let k_normed = shared_k[kd] * k_norm_scale;
      let state_idx = recurrent_head_base + kd * head_v_dim + lane;
      recurrent_state[state_idx] = recurrent_state[state_idx] + k_normed * delta;
    }
  }

  var out_value = 0.0;
  if (is_v_active) {
    for (var kd: u32 = 0u; kd < head_k_dim; kd = kd + 1u) {
      let q_normed = shared_q[kd] * q_norm_scale;
      let state_idx = recurrent_head_base + kd * head_v_dim + lane;
      out_value = out_value + recurrent_state[state_idx] * q_normed;
    }
  }

  let out_sum_sq = reduce_sum(select(0.0, out_value * out_value, is_v_active), lane);
  let inv_rms = inverseSqrt(out_sum_sq / f32(head_v_dim) + params.rms_norm_eps);

  if (is_v_active) {
    let z_row_base = head * head_v_dim;
    let z_packed_base = params.conv_dim + z_row_base;
    let z_index = select(z_row_base + lane, z_packed_base + lane, (params.packed_flags & 2u) != 0u);
    let gate = silu(f32(z_proj[z_index]));
    let norm_index = select(lane, head * head_v_dim + lane, params.norm_mode == 1u);
    output[out_row_base + lane] = (out_value * inv_rms) * norm_weight[norm_index] * gate;
  }
}

