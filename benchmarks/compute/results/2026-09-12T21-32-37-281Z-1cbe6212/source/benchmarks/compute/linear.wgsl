// Controlled scalar matmul fixture: both lanes use this exact loop and layout.
// This is not a tuned production GEMM and cannot promote a runtime kernel.
override WORKGROUP_SIZE: u32;
override FUSE_EPILOGUE: bool;
override SIGMOID_CLAMP: f32;

struct Dimensions {
    rows: u32,
    columns: u32,
    inner: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= dims.rows * dims.columns) { return; }
    let row = index / dims.columns;
    let column = index % dims.columns;
    var value = 0.0;
    for (var inner = 0u; inner < dims.inner; inner += 1u) {
        value += input[row * dims.inner + inner] * weights[inner * dims.columns + column];
    }
    if (FUSE_EPILOGUE) {
        value += bias[column];
        value *= 1.0 / (1.0 + exp(-clamp(value, -SIGMOID_CLAMP, SIGMOID_CLAMP)));
    }
    output[index] = value;
}
