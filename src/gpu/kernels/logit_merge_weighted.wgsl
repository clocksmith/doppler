override WORKGROUP_SIZE: u32 = 256u;

struct MergeParams {
  vocab_size: u32,
  weight_a: f32,
  weight_b: f32,
  temperature: f32,
}

@group(0) @binding(0) var<storage, read> logits_a: array<f32>;
@group(0) @binding(1) var<storage, read> logits_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> merged: array<f32>;
@group(0) @binding(3) var<uniform> params: MergeParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.vocab_size) {
    return;
  }
  let weighted = params.weight_a * logits_a[index] + params.weight_b * logits_b[index];
  merged[index] = select(weighted, weighted / params.temperature, params.temperature != 1.0);
}
