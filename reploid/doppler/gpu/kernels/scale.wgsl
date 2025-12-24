/**
 * Scale kernel - multiply each element by a scalar factor
 * Used for embedding scaling in Gemma models (sqrt(hidden_size))
 */

struct Uniforms {
  count: u32,
  scale: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uniforms.count) {
    return;
  }
  output[idx] = input[idx] * uniforms.scale;
}

// In-place variant (input and output are same buffer)
@compute @workgroup_size(256, 1, 1)
fn main_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uniforms.count) {
    return;
  }
  output[idx] = output[idx] * uniforms.scale;
}
