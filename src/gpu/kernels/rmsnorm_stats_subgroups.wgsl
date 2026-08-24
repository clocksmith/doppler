enable subgroups;

override WORKGROUP_SIZE: u32 = 256u;
const MAX_SUBGROUPS: u32 = 32u;

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

var<workgroup> subgroup_partial_sums: array<f32, MAX_SUBGROUPS>;

fn reduce_sum(
  local_sum_sq: f32,
  thread_index: u32,
  subgroup_lane: u32,
  subgroup_size: u32
) -> f32 {
  let subgroup_id = thread_index / subgroup_size;
  let subgroup_count = (WORKGROUP_SIZE + subgroup_size - 1u) / subgroup_size;
  let subgroup_sum = subgroupAdd(local_sum_sq);
  if (subgroup_lane == 0u && subgroup_id < subgroup_count) {
    subgroup_partial_sums[subgroup_id] = subgroup_sum;
  }
  workgroupBarrier();
  if (thread_index == 0u) {
    var sum = 0.0;
    for (var subgroup = 0u; subgroup < subgroup_count; subgroup = subgroup + 1u) {
      sum = sum + subgroup_partial_sums[subgroup];
    }
    subgroup_partial_sums[0] = sum;
  }
  workgroupBarrier();
  return subgroup_partial_sums[0];
}

fn token_index(workgroup_id: vec3<u32>) -> u32 {
  return workgroup_id.y * max(params.token_stride, 1u) + workgroup_id.x;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_size) subgroup_size: u32
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
  let total_sum = reduce_sum(local_sum_sq, thread_index, subgroup_lane, subgroup_size);
  if (thread_index == 0u) {
    inv_rms[token] = inverseSqrt(total_sum / f32(params.hidden_size) + params.eps);
  }
}
