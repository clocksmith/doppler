// MXFP4 Dequantization Kernel
//
// Dequantizes MXFP4 quantized weights used by GPT-OSS models.
//
// MXFP4 Format (per group of 32 elements):
// - blocks: 16 bytes containing 32 x 4-bit values (2 nibbles per byte)
// - scale: 1 byte shared across the 32 values
//
// Tensor shapes:
// - blocks: [..., num_groups, 16] - U8 packed nibbles
// - scales: [..., num_groups] - U8 scale factors
//
// Dequantization formula:
//   value = (nibble - 8) * scale * (1.0 / 127.0)
//
// Where nibble is 0-15 (representing -8 to +7 signed 4-bit)

struct Uniforms {
    total_elements: u32,    // Total output elements
    num_groups: u32,        // Groups per row (e.g., 90)
    group_size: u32,        // Elements per group (32 = 16 bytes * 2 nibbles)
    row_stride: u32,        // Stride between rows in output
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> blocks: array<u32>;  // Packed U8 as U32
@group(0) @binding(2) var<storage, read> scales: array<u32>;  // Packed U8 as U32
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Workgroup shared memory for scales
var<workgroup> shared_scale: f32;

// Extract 4-bit nibble from packed bytes
fn get_nibble(byte_data: u32, nibble_idx: u32) -> i32 {
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

    // Convert to signed: 0-15 -> -8 to +7
    return i32(nibble) - 8;
}

// Get scale value from packed scales array
fn get_scale(scale_data: u32, idx: u32) -> f32 {
    let byte_idx = idx % 4u;
    let scale_byte = (scale_data >> (byte_idx * 8u)) & 0xFFu;
    // Normalize scale: U8 [0, 255] -> [-1, 1] range approximately
    // MXFP4 typically uses scale as a simple multiplier
    return f32(scale_byte) / 127.0;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let elem_idx = global_id.x;

    if (elem_idx >= uniforms.total_elements) {
        return;
    }

    // Compute which group this element belongs to
    let group_idx = elem_idx / uniforms.group_size;
    let intra_group_idx = elem_idx % uniforms.group_size;

    // Get scale for this group
    let scale_word_idx = group_idx / 4u;
    let scale_byte_idx = group_idx % 4u;
    let scale_word = scales[scale_word_idx];
    let scale = get_scale(scale_word, scale_byte_idx);

    // Get the block data
    // Each group has 16 bytes = 4 U32 words = 32 nibbles
    let block_base = group_idx * 4u;  // 4 U32 words per group
    let word_in_block = intra_group_idx / 8u;  // Which U32 word (0-3)
    let nibble_in_word = intra_group_idx % 8u;  // Which nibble in word (0-7)

    let block_word = blocks[block_base + word_in_block];
    let nibble_val = get_nibble(block_word, nibble_in_word);

    // Dequantize: signed nibble * scale
    let dequant = f32(nibble_val) * scale;

    output[elem_idx] = dequant;
}

// Vectorized version - each thread handles 4 elements
@compute @workgroup_size(64, 1, 1)
fn main_vec4(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let base_elem = global_id.x * 4u;

    if (base_elem >= uniforms.total_elements) {
        return;
    }

    // Compute group for base element
    let group_idx = base_elem / uniforms.group_size;
    let intra_group_base = base_elem % uniforms.group_size;

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
        if (elem_idx >= uniforms.total_elements) {
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

// Expert-aware version: dequantizes a single expert's slice from packed tensor
// Input tensors have shape [num_experts, out_dim, num_groups, 16]
// This kernel extracts and dequantizes a single expert's weights
struct ExpertUniforms {
    expert_idx: u32,        // Which expert to extract
    num_experts: u32,       // Total experts (32 for GPT-OSS)
    out_dim: u32,           // Output dimension
    num_groups: u32,        // Groups per row (90 for GPT-OSS)
    total_output: u32,      // Total output elements for this expert
}

@group(0) @binding(0) var<uniform> expert_uniforms: ExpertUniforms;
@group(0) @binding(1) var<storage, read> expert_blocks: array<u32>;
@group(0) @binding(2) var<storage, read> expert_scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> expert_output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main_expert(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let out_elem = global_id.x;

    if (out_elem >= expert_uniforms.total_output) {
        return;
    }

    // Output layout: [out_dim, group_size * num_groups]
    // For out_dim=5760, num_groups=90, group_size=32: output is [5760, 2880]
    let row_idx = out_elem / (expert_uniforms.num_groups * 32u);
    let col_idx = out_elem % (expert_uniforms.num_groups * 32u);
    let group_in_row = col_idx / 32u;
    let elem_in_group = col_idx % 32u;

    // Input blocks layout: [num_experts, out_dim, num_groups, 16] as U8
    // = [num_experts, out_dim, num_groups, 4] as U32
    let expert_offset = expert_uniforms.expert_idx;
    let blocks_per_expert = expert_uniforms.out_dim * expert_uniforms.num_groups * 4u;
    let blocks_per_row = expert_uniforms.num_groups * 4u;

    let block_word_idx = expert_offset * blocks_per_expert
                       + row_idx * blocks_per_row
                       + group_in_row * 4u
                       + (elem_in_group / 8u);

    let block_word = expert_blocks[block_word_idx];
    let nibble_in_word = elem_in_group % 8u;
    let nibble_val = get_nibble(block_word, nibble_in_word);

    // Input scales layout: [num_experts, out_dim, num_groups] as U8
    // = [num_experts, out_dim, ceil(num_groups/4)] as U32
    let scales_per_expert = expert_uniforms.out_dim * ((expert_uniforms.num_groups + 3u) / 4u);
    let scales_per_row = (expert_uniforms.num_groups + 3u) / 4u;

    let scale_word_idx = expert_offset * scales_per_expert
                       + row_idx * scales_per_row
                       + (group_in_row / 4u);
    let scale_word = expert_scales[scale_word_idx];
    let scale = get_scale(scale_word, group_in_row % 4u);

    expert_output[out_elem] = f32(nibble_val) * scale;
}
