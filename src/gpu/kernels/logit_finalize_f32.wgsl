override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    row_count: u32,
    source_columns: u32,
    target_columns: u32,
    has_bias: u32,
    output_scale: f32,
    softcap: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let count = u.row_count * u.target_columns;
    if (index >= count) {
        return;
    }
    let row = index / u.target_columns;
    let column = index % u.target_columns;
    if (column >= u.source_columns) {
        output[index] = bitcast<f32>(0xff800000u);
        return;
    }
    var value = input[row * u.source_columns + column];
    if (u.has_bias == 1u) {
        value = value + bias[column];
    }
    value = value * u.output_scale;
    if (u.softcap > 0.0) {
        value = u.softcap * tanh(value / u.softcap);
    }
    output[index] = value;
}
