// MXFP4 Dequantization Kernel (expert slice)
//
// Expert-aware version: dequantizes a single expert's slice from packed tensor.

// Tunable workgroup size
override WORKGROUP_SIZE_EXPERT: u32 = 256u;

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

// Expert-aware version: dequantizes a single expert's slice from packed tensor
// Input tensors have shape [num_experts, out_dim, num_groups, 16]
// This kernel extracts and dequantizes a single expert's weights
struct ExpertUniforms {
    expert_idx: u32,        // Which expert to extract
    num_experts: u32,       // Total experts (32 for GPT-OSS)
    out_dim: u32,           // Output dimension
    num_groups: u32,        // Groups per row (90 for GPT-OSS)
    total_output: u32,      // Total output elements for this expert
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> eu: ExpertUniforms;
@group(0) @binding(1) var<storage, read> expert_blocks: array<u32>;
@group(0) @binding(2) var<storage, read> expert_scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> expert_output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE_EXPERT, 1, 1)
fn main_expert(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let out_elem = global_id.x;

    if (out_elem >= eu.total_output) {
        return;
    }

    // Output layout: [out_dim, group_size * num_groups]
    // For out_dim=5760, num_groups=90, group_size=32: output is [5760, 2880]
    let row_idx = out_elem / (eu.num_groups * 32u);
    let col_idx = out_elem % (eu.num_groups * 32u);
    let group_in_row = col_idx / 32u;
    let elem_in_group = col_idx % 32u;

    // Input blocks layout: [num_experts, out_dim, num_groups, 16] as U8
    // = [num_experts, out_dim, num_groups, 4] as U32
    let expert_offset = eu.expert_idx;
    let blocks_per_expert = eu.out_dim * eu.num_groups * 4u;
    let blocks_per_row = eu.num_groups * 4u;

    let block_word_idx = expert_offset * blocks_per_expert
                       + row_idx * blocks_per_row
                       + group_in_row * 4u
                       + (elem_in_group / 8u);

    let block_word = expert_blocks[block_word_idx];
    let nibble_in_word = elem_in_group % 8u;
    let nibble_val = get_nibble(block_word, nibble_in_word);

    // Input scales layout: [num_experts, out_dim, num_groups] as contiguous U8 bytes
    // Calculate byte offset, then convert to U32 word index
    let scales_bytes_per_expert = eu.out_dim * eu.num_groups;
    let scales_bytes_per_row = eu.num_groups;

    let scale_byte_offset = expert_offset * scales_bytes_per_expert
                          + row_idx * scales_bytes_per_row
                          + group_in_row;
    let scale_word_idx = scale_byte_offset / 4u;
    let scale_byte_in_word = scale_byte_offset % 4u;
    let scale_word = expert_scales[scale_word_idx];
    let scale = get_scale(scale_word, scale_byte_in_word);

    expert_output[out_elem] = f32(nibble_val) * scale;
}
