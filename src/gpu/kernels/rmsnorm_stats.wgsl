override WORKGROUP_SIZE: u32 = 256u;

struct RMSNormStatsParams {
  hidden_size: u32,
  num_tokens: u32,
  eps: f32,
  token_stride: u32,
}

@group(0) @binding(0) var<uniform> params: RMSNormStatsParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> prenorm_sum: array<f32>;
@group(0) @binding(4) var<storage, read_write> inv_rms: array<f32>;

var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;

fn reduce_sum(local_sum_sq: f32, thread_index: u32) -> f32 {
  shared_sum[thread_index] = local_sum_sq;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (thread_index < stride) {
      shared_sum[thread_index] = shared_sum[thread_index] + shared_sum[thread_index + stride];
    }
    workgroupBarrier();
  }
  return shared_sum[0];
}

fn token_index(workgroup_id: vec3<u32>) -> u32 {
  return workgroup_id.y * max(params.token_stride, 1u) + workgroup_id.x;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let token = token_index(workgroup_id);
  let thread_index = local_id.x;
  if (token >= params.num_tokens) {
    return;
  }
  let base = token * params.hidden_size;
  let elements_per_thread = (params.hidden_size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
  var local_sum_sq = 0.0;
  for (var element = 0u; element < elements_per_thread; element = element + 1u) {
    let index = thread_index * elements_per_thread + element;
    if (index < params.hidden_size) {
      let value = input[base + index] + residual[base + index];
      prenorm_sum[base + index] = value;
      local_sum_sq = local_sum_sq + value * value;
    }
  }
  let total_sum = reduce_sum(local_sum_sq, thread_index);
  if (thread_index == 0u) {
    inv_rms[token] = inverseSqrt(total_sum / f32(params.hidden_size) + params.eps);
  }
}
