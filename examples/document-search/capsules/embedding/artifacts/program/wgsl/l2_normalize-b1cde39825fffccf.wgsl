override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    row_count: u32,
    hidden_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row = wid.x;
    let local = lid.x;
    if (row >= u.row_count) {
        return;
    }
    let base = row * u.hidden_size;
    var sum = 0.0;
    var column = local;
    while (column < u.hidden_size) {
        let value = input[base + column];
        sum = sum + value * value;
        column = column + WORKGROUP_SIZE;
    }
    partial[local] = sum;
    workgroupBarrier();
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (local < stride) {
            partial[local] = partial[local] + partial[local + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    var inverse_norm = 0.0;
    if (partial[0] > 0.0) {
        inverse_norm = inverseSqrt(partial[0]);
    }
    column = local;
    while (column < u.hidden_size) {
        output[base + column] = input[base + column] * inverse_norm;
        column = column + WORKGROUP_SIZE;
    }
}
