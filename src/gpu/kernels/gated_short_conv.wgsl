override WORKGROUP_SIZE: u32 = 256u;

struct Params {
  num_tokens: u32,
  hidden_size: u32,
  kernel_size: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> conv_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> conv_state: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let channel = gid.x;
  if (channel >= params.hidden_size) {
    return;
  }
  let hidden_size = params.hidden_size;
  let kernel_size = params.kernel_size;
  let state_width = kernel_size - 1u;
  let row_stride = 3u * hidden_size;
  let state_base = channel * state_width;
  let weight_base = channel * kernel_size;
  for (var token: u32 = 0u; token < params.num_tokens; token = token + 1u) {
    let row_offset = token * row_stride;
    let b_value = input[row_offset + channel];
    let c_value = input[row_offset + hidden_size + channel];
    let x_value = input[row_offset + 2u * hidden_size + channel];
    let bx = b_value * x_value;
    var conv_sum: f32 = 0.0;
    for (var kernel_index: u32 = 0u; kernel_index < state_width; kernel_index = kernel_index + 1u) {
      conv_sum = conv_sum + conv_state[state_base + kernel_index] * conv_weight[weight_base + kernel_index];
    }
    conv_sum = conv_sum + bx * conv_weight[weight_base + state_width];
    for (var kernel_index: u32 = 0u; kernel_index + 1u < state_width; kernel_index = kernel_index + 1u) {
      conv_state[state_base + kernel_index] = conv_state[state_base + kernel_index + 1u];
    }
    if (state_width > 0u) {
      conv_state[state_base + state_width - 1u] = bx;
    }
    output[token * hidden_size + channel] = c_value * conv_sum;
  }
}
