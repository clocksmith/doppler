// vision_rope_2d.wgsl

override WORKGROUP_SIZE: u32 = 256u;

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

    let pairs_per_axis = u.head_dim / 4u;
    let pairs_per_head = pairs_per_axis * 2u;
    let token_idx = idx / (u.num_heads * pairs_per_head);
    let head_pair = idx % (u.num_heads * pairs_per_head);
    let head_idx = head_pair / pairs_per_head;
    let axis_pair = head_pair % pairs_per_head;
    let is_y_axis = axis_pair >= pairs_per_axis;
    let pair_idx = axis_pair % pairs_per_axis;
    let position = select(token_idx % u.grid_width, token_idx / u.grid_width, is_y_axis);
    let spatial_dim = u.head_dim / 2u;
    let exponent = f32(pair_idx * 2u) / f32(spatial_dim);
    let angle = f32(position) / pow(u.rope_theta, exponent);
    let cos_value = cos(angle);
    let sin_value = sin(angle);
    let axis_offset = select(0u, spatial_dim, is_y_axis);
    let base_idx = token_idx * u.num_heads * u.head_dim + head_idx * u.head_dim + axis_offset;
    let first_idx = base_idx + pair_idx;
    let second_idx = first_idx + pairs_per_axis;
    let first = input[first_idx];
    let second = input[second_idx];
    input[first_idx] = first * cos_value - second * sin_value;
    input[second_idx] = second * cos_value + first * sin_value;
}
