// MXFP4 Dequantization Kernel (vec4)
//
// Vectorized variant for MXFP4 dequantization.

// Tunable workgroup size
override WORKGROUP_SIZE_VEC4: u32 = 64u;

struct Uniforms {
    total_elements: u32,    // Total output elements
    num_groups: u32,        // Groups per row (e.g., 90)
    group_size: u32,        // Elements per group (32 = 16 bytes * 2 nibbles)
    row_stride: u32,        // Stride between rows in output
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> blocks: array<u32>;  // Packed U8 as U32
@group(0) @binding(2) var<storage, read> scales: array<u32>;  // Packed U8 as U32
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Extract 4-bit nibble from packed bytes and decode as E2M1 FP4
// MXFP4 E2M1 format: 1 sign bit, 2 exponent bits (bias=1), 1 mantissa bit
// Layout: S | E1 | E0 | M
fn get_nibble(byte_data: u32, nibble_idx: u32) -> f32 {
    // Each U32 contains 4 bytes, each byte contains 2 nibbles
    let byte_idx = nibble_idx / 2u;
    let is_high = nibble_idx % 2u;
    let byte_val = (byte_data >> (byte_idx * 8u)) & 0xFFu;

    var nibble: u32;
    if (is_high == 1u) {
        nibble = (byte_val >> 4u) & 0xFu;
    } else {
        nibble = byte_val & 0xFu;
    }

    // Decode E2M1 FP4:
    // nibble = SEEM (4 bits)
    // S = sign bit (bit 3)
    // E = exponent (bits 2-1), bias = 1
    // M = mantissa (bit 0)
    let sign_bit = (nibble >> 3u) & 1u;
    let exp = (nibble >> 1u) & 3u;  // 2-bit exponent
    let mantissa = nibble & 1u;     // 1-bit mantissa

    var value: f32;
    if (exp == 0u) {
        // Subnormal: value = (-1)^S * 0.5 * M
        value = f32(mantissa) * 0.5;
    } else {
        // Normal: value = (-1)^S * (1 + 0.5*M) * 2^(E-1)
        let m = 1.0 + f32(mantissa) * 0.5;
        value = m * pow(2.0, f32(exp) - 1.0);
    }

    // Apply sign
    if (sign_bit == 1u) {
        value = -value;
    }
    return value;
}

// Get scale value from packed scales array (E8M0 format)
fn get_scale(scale_data: u32, idx: u32) -> f32 {
    let byte_idx = idx % 4u;
    let scale_byte = (scale_data >> (byte_idx * 8u)) & 0xFFu;
    // E8M0 format: 8-bit exponent, no mantissa
    // scale = 2^(exponent - 127) where 127 is the IEEE bias
    // Special cases: 0 = zero, 255 = NaN (we treat as 0)
    if (scale_byte == 0u || scale_byte == 255u) {
        return 0.0;
    }
    let exponent = i32(scale_byte) - 127;
    return pow(2.0, f32(exponent));
}

// Vectorized version - each thread handles 4 elements
@compute @workgroup_size(WORKGROUP_SIZE_VEC4, 1, 1)
fn main_vec4(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let base_elem = global_id.x * 4u;

    if (base_elem >= u.total_elements) {
        return;
    }

    // Compute group for base element
    let group_idx = base_elem / u.group_size;
    let intra_group_base = base_elem % u.group_size;

    // Get scale for this group (all 4 elements should be in same group typically)
    let scale_word_idx = group_idx / 4u;
    let scale_byte_idx = group_idx % 4u;
    let scale_word = scales[scale_word_idx];
    let scale = get_scale(scale_word, scale_byte_idx);

    // Get block data
    let block_base = group_idx * 4u;

    // Process 4 consecutive nibbles
    for (var i = 0u; i < 4u; i = i + 1u) {
        let elem_idx = base_elem + i;
        if (elem_idx >= u.total_elements) {
            break;
        }

        let intra_group_idx = intra_group_base + i;
        let word_in_block = intra_group_idx / 8u;
        let nibble_in_word = intra_group_idx % 8u;

        let block_word = blocks[block_base + word_in_block];
        let nibble_val = get_nibble(block_word, nibble_in_word);

        output[elem_idx] = f32(nibble_val) * scale;
    }
}
