// One pass over original content, with separate instruction/complement means.
override WORKGROUP_SIZE: u32 = 256u;
struct Uniforms {
  channels: u32, start: u32, stop: u32, content_start: u32,
  content_stop: u32, _pad0: u32, _pad1: u32, _pad2: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> instruction: array<f32>;
@group(0) @binding(3) var<storage, read_write> reference: array<f32>;
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let channel = id.x;
  if (channel >= u.channels) { return; }
  var instruction_sum = 0.0;
  var reference_sum = 0.0;
  for (var token = u.content_start; token < u.content_stop; token += 1u) {
    let value = input[token * u.channels + channel];
    let energy = value * value;
    if (token >= u.start && token < u.stop) { instruction_sum += energy; }
    else { reference_sum += energy; }
  }
  let instruction_count = u.stop - u.start;
  let reference_count = u.content_stop - u.content_start - instruction_count;
  instruction[channel] = instruction_sum / f32(instruction_count);
  reference[channel] = reference_sum / f32(reference_count);
}
