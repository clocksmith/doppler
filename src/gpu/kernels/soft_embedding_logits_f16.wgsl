enable f16;

override WORKGROUP_SIZE: u32 = 256u;
const NEG_INF: f32 = -3.402823e+38;

struct Uniforms {
  canvas_length: u32,
  hidden_size: u32,
  vocab_size: u32,
  row_start: u32,
  row_count: u32,
  temperature: f32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> row_max: array<f32>;
@group(0) @binding(3) var<storage, read_write> row_sum: array<f32>;
@group(0) @binding(4) var<storage, read_write> probabilities: array<f32>;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn logits_norm_stats(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let row = workgroup_id.x;
  let thread_idx = local_id.x;
  if (row >= u.canvas_length) {
    return;
  }

  let base = row * u.vocab_size;
  let temperature = max(u.temperature, 0.000001);

  var local_max = NEG_INF;
  var idx = thread_idx;
  while (idx < u.vocab_size) {
    local_max = max(local_max, logits[base + idx] / temperature);
    idx = idx + WORKGROUP_SIZE;
  }

  shared_max[thread_idx] = local_max;
  workgroupBarrier();

  var stride = WORKGROUP_SIZE / 2u;
  while (stride > 0u) {
    if (thread_idx < stride) {
      shared_max[thread_idx] = max(shared_max[thread_idx], shared_max[thread_idx + stride]);
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  let max_value = shared_max[0];
  var local_sum = 0.0;
  idx = thread_idx;
  while (idx < u.vocab_size) {
    let value = logits[base + idx] / temperature;
    local_sum = local_sum + exp(value - max_value);
    idx = idx + WORKGROUP_SIZE;
  }

  shared_sum[thread_idx] = local_sum;
  workgroupBarrier();

  stride = WORKGROUP_SIZE / 2u;
  while (stride > 0u) {
    if (thread_idx < stride) {
      shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (thread_idx == 0u) {
    row_max[row] = max_value;
    row_sum[row] = shared_sum[0];
  }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn logits_probability_chunk(@builtin(global_invocation_id) gid: vec3<u32>) {
  let chunk_idx = gid.x;
  let row = gid.y;
  if (row >= u.canvas_length || chunk_idx >= u.row_count) {
    return;
  }

  let vocab_idx = u.row_start + chunk_idx;
  if (vocab_idx >= u.vocab_size) {
    return;
  }

  let sum = row_sum[row];
  let temperature = max(u.temperature, 0.000001);
  let value = logits[row * u.vocab_size + vocab_idx] / temperature;
  probabilities[row * u.row_count + chunk_idx] = select(
    0.0,
    exp(value - row_max[row]) / sum,
    sum > 0.0
  );
}
