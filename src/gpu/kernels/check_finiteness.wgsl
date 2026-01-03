// Finiteness Check Kernel
//
// Detects NaN or Infinity values in an activation buffer and atomically sets a global status flag
// Used to prevent F16 overflow values from permanently poisoning the KV cache

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    size: u32,
    layer: u32,
    step: u32,
    abs_threshold: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> status: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    
    if (idx >= u.size) {
        return;
    }

    let val = f32(input[idx]);
    let bits = bitcast<u32>(val);
    
    // Check for NaN/Infinity and near-overflow magnitude before corruption.
    let non_finite = (bits & 0x7F800000u) == 0x7F800000u;
    let exceeds_abs_threshold = abs(val) > u.abs_threshold;
    if (non_finite || exceeds_abs_threshold) {
        let old = atomicCompareExchangeWeak(&status[0], 0u, 1u);
        if (old.exchanged) {
            atomicStore(&status[1], u.layer);
            atomicStore(&status[2], u.step);
        }
    }
}
