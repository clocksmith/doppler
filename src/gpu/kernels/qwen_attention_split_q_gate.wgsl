override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    element_count: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> query: array<f32>;
@group(0) @binding(3) var<storage, read_write> gate: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_index = gid.x;
    if (output_index >= u.element_count) {
        return;
    }
    let row = output_index / u.head_dim;
    let column = output_index % u.head_dim;
    let input_base = row * u.head_dim * 2u;
    query[output_index] = input[input_base + column];
    gate[output_index] = input[input_base + u.head_dim + column];
}
