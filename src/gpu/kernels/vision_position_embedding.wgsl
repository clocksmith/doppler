// vision_position_embedding.wgsl

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    grid_height: u32,
    grid_width: u32,
    position_embedding_size: u32,
    hidden_size: u32,
    output_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_idx = gid.x;
    if (output_idx >= u.output_elements) {
        return;
    }
    let token_idx = output_idx / u.hidden_size;
    let hidden_idx = output_idx % u.hidden_size;
    let x = token_idx % u.grid_width;
    let y = token_idx / u.grid_width;
    let y_table_offset = u.position_embedding_size * u.hidden_size;
    output[output_idx] = table[x * u.hidden_size + hidden_idx]
        + table[y_table_offset + y * u.hidden_size + hidden_idx];
}
