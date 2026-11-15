// One workgroup rotates each token in the frozen rank-two plane.
override WORKGROUP_SIZE: u32 = 256u;
struct Uniforms { channels: u32, tokens: u32, theta: f32, _pad0: u32, }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> basis: array<f32>;
@group(0) @binding(3) var<storage, read> status: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
var<workgroup> sums: array<vec2<f32>, WORKGROUP_SIZE>;
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(local_invocation_id) id: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let lane = id.x;
  let token = group.x;
  if (token >= u.tokens) { return; }
  let offset = token * u.channels;
  var dots = vec2<f32>(0.0);
  for (var channel = lane; channel < u.channels; channel += WORKGROUP_SIZE) {
    dots += input[offset + channel] * vec2<f32>(basis[channel], basis[u.channels + channel]);
  }
  sums[lane] = dots;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  let cosine_delta = cos(u.theta) - 1.0;
  let sine = sin(u.theta);
  let first = cosine_delta * sums[0].x - sine * sums[0].y;
  let second = cosine_delta * sums[0].y + sine * sums[0].x;
  for (var channel = lane; channel < u.channels; channel += WORKGROUP_SIZE) {
    var value = input[offset + channel];
    if (status[0] < 0.0) { value = 0.0; }
    else if (status[0] > 0.0 && u.theta != 0.0) {
      value += first * basis[channel] + second * basis[u.channels + channel];
    }
    output[offset + channel] = value;
  }
}
