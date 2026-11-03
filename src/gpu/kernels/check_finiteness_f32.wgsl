// AUTO-GENERATED from src/gpu/kernels/check_finiteness.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
  size: u32,
  layer: u32,
  step: u32,
  abs_threshold: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> status: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= u.size) {
    return;
  }
  let value = input[idx];
  let bits = bitcast<u32>(value);
  let non_finite = (bits & 0x7F800000u) == 0x7F800000u;
  let exceeds_abs_threshold = abs(value) > u.abs_threshold;
  if (non_finite || exceeds_abs_threshold) {
    let old = atomicOr(&status[0], 1u);
    if (old == 0u) {
      atomicStore(&status[1], u.layer);
      atomicStore(&status[2], u.step);
    }
  }
}
