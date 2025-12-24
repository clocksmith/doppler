// BF16 to F32 Conversion Kernel
//
// Converts BF16 (bfloat16) data to F32.
// BF16 is just the upper 16 bits of F32, so conversion is a simple shift.
//
// Supports 2D dispatch for large tensors (>65535 workgroups).

const WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUPS_X: u32 = 65535u;

struct Uniforms {
    numElements: u32,
    inputOffset: u32,   // Element offset for input (in BF16 elements)
    outputOffset: u32,  // Element offset for output (in F32 elements)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<u32>;  // BF16 packed as u32 (2 per u32)
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Support 2D dispatch for large tensors (>65535*256 elements)
    // When using 2D dispatch: linear_idx = x + y * MAX_WORKGROUPS_X * WORKGROUP_SIZE
    let linear_idx = global_id.x + global_id.y * MAX_WORKGROUPS_X * WORKGROUP_SIZE;

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

    // Extract two BF16 values and convert to F32
    // BF16 is upper 16 bits of F32, so shift left by 16
    let bf16_lo = packed & 0xFFFFu;
    let bf16_hi = (packed >> 16u) & 0xFFFFu;

    // Convert by shifting to F32 position
    output[output_elem_idx] = bitcast<f32>(bf16_lo << 16u);

    if (local_elem_idx + 1u < uniforms.numElements) {
        output[output_elem_idx + 1u] = bitcast<f32>(bf16_hi << 16u);
    }
}

// Single-element version for odd-sized tensors
@compute @workgroup_size(256, 1, 1)
fn main_single(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= uniforms.numElements) {
        return;
    }

    // Read as u16 pairs, extract the right one
    let pair_idx = idx / 2u;
    let packed = input[pair_idx];

    let bf16 = select(packed & 0xFFFFu, (packed >> 16u) & 0xFFFFu, (idx & 1u) == 1u);
    output[idx] = bitcast<f32>(bf16 << 16u);
}
