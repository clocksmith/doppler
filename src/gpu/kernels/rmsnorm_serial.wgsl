// Sequential per-row reduction: explicit source-f32 qualification lane.
override WORKGROUP_SIZE: u32 = 64u;
override ROWS: u32 = 33u;
override HIDDEN_SIZE: u32 = 256u;
override EPSILON: f32 = 0.000001;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= ROWS) { return; }
    var squares = 0.0;
    for (var d = 0u; d < HIDDEN_SIZE; d++) {
        let value = input[row * HIDDEN_SIZE + d]; squares += value * value;
    }
    let inv = inverseSqrt(squares / f32(HIDDEN_SIZE) + EPSILON);
    for (var d = 0u; d < HIDDEN_SIZE; d++) { output[row * HIDDEN_SIZE + d] = input[row * HIDDEN_SIZE + d] * inv * weight[d]; }
}
