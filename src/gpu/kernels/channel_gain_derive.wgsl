// A single workgroup centers the clipped log gains over all channels.
override WORKGROUP_SIZE: u32 = 256u;
struct Uniforms { channels: u32, beta: f32, clip: f32, epsilon: f32, }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> instruction: array<f32>;
@group(0) @binding(2) var<storage, read> reference: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> invalid: atomic<u32>;
var<workgroup> sums: array<f32, WORKGROUP_SIZE>;
fn log_gain(channel: u32) -> f32 {
  let a = instruction[channel];
  let b = reference[channel];
  if ((bitcast<u32>(a) & 0x7f800000u) == 0x7f800000u ||
      (bitcast<u32>(b) & 0x7f800000u) == 0x7f800000u || a < 0.0 || b < 0.0) {
    atomicOr(&invalid, 1u);
    return 0.0;
  }
  return u.beta * clamp((a - b) / (b + u.epsilon), -u.clip, u.clip);
}
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(local_invocation_id) id: vec3<u32>) {
  let lane = id.x;
  if (lane == 0u) { atomicStore(&invalid, 0u); }
  workgroupBarrier();
  var sum = 0.0;
  for (var channel = lane; channel < u.channels; channel += WORKGROUP_SIZE) {
    sum += log_gain(channel);
  }
  sums[lane] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  let mean = sums[0] / f32(u.channels);
  for (var channel = lane; channel < u.channels; channel += WORKGROUP_SIZE) {
    let gain = exp(log_gain(channel) - mean);
    output[channel] = select(gain, 0.0, atomicLoad(&invalid) != 0u);
  }
}
