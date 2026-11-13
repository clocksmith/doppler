override WORKGROUP_SIZE: u32 = 256u;
struct Uniforms { channels: u32, _pad0: u32, _pad1: u32, _pad2: u32, }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> gains: array<f32>;
@group(0) @binding(2) var<storage, read> seeds: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> sums: array<vec2<f32>, WORKGROUP_SIZE>;
var<workgroup> invalid: atomic<u32>;
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn permuted(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u.channels) { return; }
  let source = seeds[id.x];
  if (source >= u.channels) { output[id.x] = 0.0; return; }
  output[id.x] = gains[source];
}
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn random_z(@builtin(local_invocation_id) id: vec3<u32>) {
  let lane = id.x;
  if (lane == 0u) { atomicStore(&invalid, 0u); }
  workgroupBarrier();
  var local = vec2<f32>(0.0);
  for (var channel = lane; channel < u.channels; channel += WORKGROUP_SIZE) {
    let g = gains[channel];
    let z = bitcast<f32>(seeds[channel]);
    if (g <= 0.0 || (bitcast<u32>(g) & 0x7f800000u) == 0x7f800000u ||
        (seeds[channel] & 0x7f800000u) == 0x7f800000u) { atomicOr(&invalid, 1u); }
    let log_g = log(g);
    local += vec2<f32>(z, log_g * log_g);
  }
  sums[lane] = local;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  let mean_z = sums[0].x / f32(u.channels);
  let energy_g = sums[0].y;
  var centered_energy = 0.0;
  for (var channel = lane; channel < u.channels; channel += WORKGROUP_SIZE) {
    let centered = bitcast<f32>(seeds[channel]) - mean_z;
    centered_energy += centered * centered;
  }
  workgroupBarrier();
  sums[lane] = vec2<f32>(centered_energy, 0.0);
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  let energy_z = sums[0].x;
  for (var channel = lane; channel < u.channels; channel += WORKGROUP_SIZE) {
    if (atomicLoad(&invalid) != 0u || (energy_z == 0.0 && energy_g > 0.0)) {
      output[channel] = 0.0;
    } else if (energy_g == 0.0) {
      output[channel] = 1.0;
    } else {
      output[channel] = exp((bitcast<f32>(seeds[channel]) - mean_z) * sqrt(energy_g / energy_z));
    }
  }
}
