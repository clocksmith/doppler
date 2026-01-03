// gptoss_mxfp4_expert_fused.wgsl

/**
 * GPT-OSS MXFP4 expert dequant kernel with vec4-oriented write path.
 *
 * This keeps the same bind-group contract as dequant_mxfp4_expert_f16.wgsl,
 * but emits f16 and processes contiguous groups to reduce scalar indexing
 * overhead on subgroup-capable backends.
 */

enable f16;

override WORKGROUP_SIZE_EXPERT: u32 = 256u;

struct ExpertUniforms {
    expert_idx: u32,
    num_experts: u32,
    out_dim: u32,
    num_groups: u32,
    total_output: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> eu: ExpertUniforms;
@group(0) @binding(1) var<storage, read> expert_blocks: array<u32>;
@group(0) @binding(2) var<storage, read> expert_scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> expert_output: array<f16>;

fn unpack_nibble(byte_data: u32, nibble_idx: u32) -> f32 {
    let byte_idx = nibble_idx / 2u;
    let is_high = nibble_idx % 2u;
    let byte_val = (byte_data >> (byte_idx * 8u)) & 0xFFu;

    var nibble: u32;
    if (is_high == 1u) {
        nibble = (byte_val >> 4u) & 0xFu;
    } else {
        nibble = byte_val & 0xFu;
    }

    let sign_bit = (nibble >> 3u) & 1u;
    let exp = (nibble >> 1u) & 3u;
    let mantissa = nibble & 1u;

    var value: f32;
    if (exp == 0u) {
        value = f32(mantissa) * 0.5;
    } else {
        let m = 1.0 + f32(mantissa) * 0.5;
        value = m * pow(2.0, f32(exp) - 1.0);
    }

    if (sign_bit == 1u) {
        value = -value;
    }
    return value;
}

fn read_scale(scale_data: u32, idx: u32) -> f32 {
    let byte_idx = idx % 4u;
    let scale_byte = (scale_data >> (byte_idx * 8u)) & 0xFFu;
    if (scale_byte == 0u || scale_byte == 255u) {
        return 0.0;
    }
    return exp2(f32(i32(scale_byte) - 127));
}

fn compute_output_value(out_elem: u32) -> f16 {
    let row_idx = out_elem / (eu.num_groups * 32u);
    let col_idx = out_elem % (eu.num_groups * 32u);
    let group_in_row = col_idx / 32u;
    let elem_in_group = col_idx % 32u;

    let expert_offset = eu.expert_idx;
    let blocks_per_expert = eu.out_dim * eu.num_groups * 4u;
    let blocks_per_row = eu.num_groups * 4u;

    let block_word_idx = expert_offset * blocks_per_expert
                       + row_idx * blocks_per_row
                       + group_in_row * 4u
                       + (elem_in_group / 8u);

    let block_word = expert_blocks[block_word_idx];
    let nibble_in_word = elem_in_group % 8u;
    let nibble_val = unpack_nibble(block_word, nibble_in_word);

    let scales_bytes_per_expert = eu.out_dim * eu.num_groups;
    let scales_bytes_per_row = eu.num_groups;
    let scale_byte_offset = expert_offset * scales_bytes_per_expert
                          + row_idx * scales_bytes_per_row
                          + group_in_row;
    let scale_word_idx = scale_byte_offset / 4u;
    let scale_byte_in_word = scale_byte_offset % 4u;
    let scale = read_scale(expert_scales[scale_word_idx], scale_byte_in_word);

    return f16(nibble_val * scale);
}

@compute @workgroup_size(WORKGROUP_SIZE_EXPERT, 1, 1)
fn main_expert(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let base = global_id.x * 4u;

    if (base >= eu.total_output) {
        return;
    }

    expert_output[base] = compute_output_value(base);

    let i1 = base + 1u;
    if (i1 < eu.total_output) {
        expert_output[i1] = compute_output_value(i1);
    }

    let i2 = base + 2u;
    if (i2 < eu.total_output) {
        expert_output[i2] = compute_output_value(i2);
    }

    let i3 = base + 3u;
    if (i3 < eu.total_output) {
        expert_output[i3] = compute_output_value(i3);
    }
}
