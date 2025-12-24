// BF16 to F16 Conversion Kernel
//
// Converts BF16 (bfloat16) data to F16.
// BF16 is represented as the upper 16 bits of F32. Conversion is:
//   bf16 -> f32 by shifting left 16 bits -> f16 cast.
//
// Supports 2D dispatch for large tensors.

enable f16;

const WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    numElements: u32,
    inputOffset: u32,   // Element offset for input (in BF16 elements)
    outputOffset: u32,  // Element offset for output (in F16 elements)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<u32>;  // BF16 packed as u32 (2 per u32)
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    // Support 2D dispatch for large tensors
    let linear_idx = global_id.x + global_id.y * num_wg.x * WORKGROUP_SIZE;

    // Each thread processes 2 BF16 values (one u32 contains 2 bf16)
    let local_pair_idx = linear_idx;
    let local_elem_idx = local_pair_idx * 2u;

    if (local_elem_idx >= uniforms.numElements) {
        return;
    }

    // Apply offsets for chunked processing
    let input_pair_idx = (uniforms.inputOffset / 2u) + local_pair_idx;
    let output_elem_idx = uniforms.outputOffset + local_elem_idx;

    let packed = input[input_pair_idx];

    let bf16_lo = packed & 0xFFFFu;
    let bf16_hi = (packed >> 16u) & 0xFFFFu;

    output[output_elem_idx] = f16(bitcast<f32>(bf16_lo << 16u));

    if (local_elem_idx + 1u < uniforms.numElements) {
        output[output_elem_idx + 1u] = f16(bitcast<f32>(bf16_hi << 16u));
    }
}
