enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    row_count: u32,
    chunk_columns: u32,
    target_columns: u32,
    column_offset: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let count = u.row_count * u.chunk_columns;
    if (index >= count) {
        return;
    }
    let row = index / u.chunk_columns;
    let column = index % u.chunk_columns;
    output[row * u.target_columns + u.column_offset + column] = f32(input[index]);
}
