// Request-local orthonormal plane from signed instruction/complement means.
override WORKGROUP_SIZE: u32 = 256u;
struct Uniforms {
  channels: u32, start: u32, stop: u32, content_start: u32,
  content_stop: u32, epsilon: f32, _pad0: u32, _pad1: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> basis: array<f32>;
@group(0) @binding(3) var<storage, read_write> status: array<f32>;
var<workgroup> sums: array<vec2<f32>, WORKGROUP_SIZE>;
var<workgroup> invalid: atomic<u32>;
fn finite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(local_invocation_id) id: vec3<u32>) {
  let lane = id.x;
  if (lane == 0u) { atomicStore(&invalid, 0u); }
  workgroupBarrier();
  var products = vec2<f32>(0.0);
  for (var channel = lane; channel < u.channels; channel += WORKGROUP_SIZE) {
    var a = 0.0;
    var b = 0.0;
    for (var token = u.content_start; token < u.content_stop; token += 1u) {
      let value = input[token * u.channels + channel];
      if (!finite(value)) { atomicOr(&invalid, 1u); }
      if (token >= u.start && token < u.stop) { a += value; }
      else { b += value; }
    }
    a /= f32(u.stop - u.start);
    b /= f32(u.content_stop - u.content_start - (u.stop - u.start));
    basis[channel] = a;
    basis[u.channels + channel] = b;
    products += vec2<f32>(a * a, a * b);
  }
  sums[lane] = products;
  workgroupBarrier();
  storageBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  let squared_a = sums[0].x;
  let norm_a = sqrt(squared_a);
  let dot_ab = sums[0].y;
  var residual_squared = 0.0;
  for (var channel = lane; channel < u.channels; channel += WORKGROUP_SIZE) {
    let first = basis[channel] / max(norm_a, u.epsilon);
    var second = basis[u.channels + channel];
    if (norm_a > u.epsilon) { second -= basis[channel] * (dot_ab / squared_a); }
    basis[channel] = first;
    basis[u.channels + channel] = second;
    residual_squared += second * second;
  }
  workgroupBarrier();
  sums[lane] = vec2<f32>(residual_squared, 0.0);
  workgroupBarrier();
  storageBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  let norm_b = sqrt(sums[0].x);
  let bad = atomicLoad(&invalid) != 0u || !finite(norm_a) || !finite(norm_b) || !finite(dot_ab);
  let degenerate = norm_a <= u.epsilon || norm_b <= u.epsilon;
  for (var channel = lane; channel < u.channels; channel += WORKGROUP_SIZE) {
    if (bad || degenerate) {
      basis[channel] = 0.0;
      basis[u.channels + channel] = 0.0;
    } else {
      basis[u.channels + channel] /= norm_b;
    }
  }
  if (lane == 0u) { status[0] = select(select(1.0, 0.0, degenerate), -1.0, bad); }
}
