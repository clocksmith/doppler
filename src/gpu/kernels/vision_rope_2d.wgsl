// vision_rope_2d.wgsl

override WORKGROUP_SIZE: u32 = 256u;
override SPATIAL_MERGE_SIZE: u32 = 1u;

struct Uniforms {
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    grid_height: u32,
    grid_width: u32,
    rope_theta: f32,
    total_pairs: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.total_pairs) {
        return;
    }

    let frequencies_per_axis = u.head_dim / 4u;
    let pairs_per_head = u.head_dim / 2u;
    let token_idx = idx / (u.num_heads * pairs_per_head);
    let head_pair = idx % (u.num_heads * pairs_per_head);
    let head_idx = head_pair / pairs_per_head;
    let first_half_dim = head_pair % pairs_per_head;
    let is_width_axis = first_half_dim >= frequencies_per_axis;
    let frequency_idx = first_half_dim % frequencies_per_axis;
    let merge_size = SPATIAL_MERGE_SIZE;
    let patches_per_block = merge_size * merge_size;
    let blocks_per_row = u.grid_width / merge_size;
    let block_idx = token_idx / patches_per_block;
    let patch_in_block = token_idx % patches_per_block;
    let block_y = block_idx / blocks_per_row;
    let block_x = block_idx % blocks_per_row;
    let local_y = patch_in_block / merge_size;
    let local_x = patch_in_block % merge_size;
    let position_x = block_x * merge_size + local_x;
    let position_y = block_y * merge_size + local_y;
    // GLM-OCR flattens [height frequencies, width frequencies], duplicates
    // that vector, then applies rotate_half across the two head halves.
    let position = select(position_y, position_x, is_width_axis);
    let exponent = f32(frequency_idx * 2u) / f32(pairs_per_head);
    let angle = f32(position) / pow(u.rope_theta, exponent);
    let cos_value = cos(angle);
    let sin_value = sin(angle);
    let base_idx = token_idx * u.num_heads * u.head_dim + head_idx * u.head_dim;
    let first_idx = base_idx + first_half_dim;
    let second_idx = first_idx + pairs_per_head;
    let first = input[first_idx];
    let second = input[second_idx];
    input[first_idx] = first * cos_value - second * sin_value;
    input[second_idx] = second * cos_value + first * sin_value;
}
