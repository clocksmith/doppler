override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    row_count: u32,
    hidden_size: u32,
    mode: u32,
    has_mask: u32,
    included_count: u32,
    last_index: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let column = gid.x;
    if (column >= u.hidden_size) {
        return;
    }
    if (u.mode == 1u) {
        output[column] = input[u.last_index * u.hidden_size + column];
        return;
    }
    var sum = 0.0;
    for (var row = 0u; row < u.row_count; row = row + 1u) {
        if (u.has_mask == 0u || mask[row] == 1u) {
            sum = sum + input[row * u.hidden_size + column];
        }
    }
    output[column] = sum / f32(u.included_count);
}
