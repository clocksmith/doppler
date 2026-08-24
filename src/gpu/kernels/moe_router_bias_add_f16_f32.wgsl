enable f16;

struct Uniforms {
  num_tokens: u32,
  num_experts: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> logits: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  let total = uniforms.num_tokens * uniforms.num_experts;
  if (index >= total) {
    return;
  }
  let expert = index % uniforms.num_experts;
  logits[index] = f16(f32(logits[index]) + bias[expert]);
}
