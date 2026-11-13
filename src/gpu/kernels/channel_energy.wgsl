// One invocation per channel; adjacent lanes read adjacent channels at each token.
override WORKGROUP_SIZE: u32 = 256u;
struct Uniforms { channels: u32, start: u32, stop: u32, _pad: u32, }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let channel = id.x;
  if (channel >= u.channels) { return; }
  var sum = 0.0;
  for (var token = u.start; token < u.stop; token += 1u) {
    let value = input[token * u.channels + channel];
    sum += value * value;
  }
  output[channel] = sum / f32(u.stop - u.start);
}
