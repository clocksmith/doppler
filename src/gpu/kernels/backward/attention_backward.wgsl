/**
 * Attention Backward Kernel (GPU)
 *
 * Computes gradients for attention outputs.
 * Placeholder implementation until full GPU backward is wired.
 */
override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    rows: u32,
    cols: u32,
    causal: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> softmax: array<f32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.rows * u.cols;
    if (idx >= total) {
        return;
    }

    let row = idx / u.cols;
    let col = idx % u.cols;
    if (u.causal != 0u && col > row) {
        output[idx] = 0.0;
        return;
    }

    var row_sum: f32 = 0.0;
    for (var j: u32 = 0u; j < u.cols; j = j + 1u) {
        if (u.causal != 0u && j > row) {
            continue;
        }
        let s = softmax[row * u.cols + j];
        row_sum = row_sum + s * grad_output[row * u.cols + j];
    }

    let s = softmax[idx];
    output[idx] = s * (grad_output[idx] - row_sum);
}
