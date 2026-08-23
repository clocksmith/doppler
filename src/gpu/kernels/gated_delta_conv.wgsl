override WORKGROUP_SIZE: u32 = 256u;

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
  _pad_u32_0: u32,
  rms_norm_eps: f32,
  qk_l2norm_eps: f32,
  packed_flags: u32,
  b_proj_offset_elements: u32,
}

@group(0) @binding(0) var<uniform> params: LinearAttentionParams;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> conv_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> conv_state: array<f32>;
@group(0) @binding(4) var<storage, read_write> conv_out: array<f32>;

fn silu(x: f32) -> f32 {
  if (x >= 0.0) {
    let z = exp(-x);
    return x / (1.0 + z);
  }
  let z = exp(x);
  return x * z / (1.0 + z);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let channel = gid.x;
  if (channel >= params.conv_dim) {
    return;
  }
  let qkvz_packed = (params.packed_flags & 2u) != 0u;
  let qkv_stride = select(params.conv_dim, params.conv_dim + params.value_dim, qkvz_packed);

  let kernel_size = params.conv_kernel_size;
  let state_base = channel * kernel_size;

  for (var token_idx: u32 = 0u; token_idx < params.num_tokens; token_idx = token_idx + 1u) {
    let qkv_idx = token_idx * qkv_stride + channel;
    let newest = f32(qkv[qkv_idx]);

    for (var k: u32 = 0u; k + 1u < kernel_size; k = k + 1u) {
      conv_state[state_base + k] = conv_state[state_base + k + 1u];
    }
    conv_state[state_base + kernel_size - 1u] = newest;

    var mixed: f32 = 0.0;
    for (var k: u32 = 0u; k < kernel_size; k = k + 1u) {
      mixed = mixed + conv_state[state_base + k] * conv_weight[state_base + k];
    }

    conv_out[token_idx * params.conv_dim + channel] = silu(mixed);
  }
}

