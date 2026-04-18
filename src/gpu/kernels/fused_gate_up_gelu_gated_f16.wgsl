// Fused Gate + Up Projection + GeGLU Activation Kernel (F16, gated)
//
// Fuses the two FFN projection matmuls with the GeGLU activation:
//   gate[m, n]  = sum_k(input[m, k] * W_gate[k, n])
//   up[m, n]    = sum_k(input[m, k] * W_up[k, n])
//   output[m, n] = gelu_tanh(gate[m, n]) * up[m, n]
//
// Each workgroup owns one row m and a column tile of the activation. Threads
// in a column group cooperate on BOTH gate and up dot products — the input
// row is read once and each loaded element feeds both matmuls, halving input
// memory traffic vs two separate fused matmul dispatches.
//
// The activation is GeLU-tanh applied to the gate branch, then multiplied by
// the up branch (GeGLU). Silu variants exist in `fused_ffn_q4k_f16.wgsl`; this
// variant ships gelu specifically for gated-gelu transformer families and has
// no runtime-selectable activation (keeping the inner loop branch-free).
//
// `intermediate_size` is a uniform — per-layer variation (e.g., double-wide
// KV-shared layers) is handled without kernel change.
//
// Dispatch: (ceil(intermediate_size / COLS_PER_WG), M, 1)
//
// Accumulation in f32 for numerical stability; output is f16 to match the
// downstream down_proj contract.
//
// No thread returns on a divergent condition mid-kernel.

enable f16;

override WORKGROUP_SIZE: u32 = 128u;
override COLS_PER_WG: u32 = 8u;

const MAX_WG: u32 = 256u;

struct Uniforms {
    M: u32,
    hidden_size: u32,
    intermediate_size: u32,
    transpose_b: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;        // [M, hidden_size]
@group(0) @binding(2) var<storage, read> W_gate: array<f16>;        // [hidden, inter] or [inter, hidden]
@group(0) @binding(3) var<storage, read> W_up: array<f16>;          // same layout as W_gate
@group(0) @binding(4) var<storage, read_write> output: array<f16>;  // [M, intermediate_size]

var<workgroup> shared_gate: array<f32, MAX_WG>;
var<workgroup> shared_up: array<f32, MAX_WG>;

fn gelu_tanh(x: f32) -> f32 {
    // GELU with tanh approximation — matches pytorch gelu_approx='tanh' and
    // Gemma's gelu_pytorch_tanh activation.
    let c: f32 = 0.7978845608;
    let k: f32 = 0.044715;
    return 0.5 * x * (1.0 + tanh(c * (x + k * x * x * x)));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let col_tile = wg.x;
    let row = wg.y;
    let tid = lid.x;
    if (row >= u.M) { return; }

    let K = u.hidden_size;
    let N = u.intermediate_size;

    let threads_per_col = WORKGROUP_SIZE / COLS_PER_WG;
    let col_within_tile = tid / threads_per_col;
    let thread_in_col = tid % threads_per_col;
    let col = col_tile * COLS_PER_WG + col_within_tile;
    let col_valid = col < N;

    var gate_partial: f32 = 0.0;
    var up_partial: f32 = 0.0;
    if (col_valid) {
        let input_row_base = row * K;
        if (u.transpose_b != 0u) {
            let w_base = col * K;
            for (var k: u32 = thread_in_col; k < K; k = k + threads_per_col) {
                let a = f32(input[input_row_base + k]);
                gate_partial = gate_partial + a * f32(W_gate[w_base + k]);
                up_partial = up_partial + a * f32(W_up[w_base + k]);
            }
        } else {
            for (var k: u32 = thread_in_col; k < K; k = k + threads_per_col) {
                let a = f32(input[input_row_base + k]);
                gate_partial = gate_partial + a * f32(W_gate[k * N + col]);
                up_partial = up_partial + a * f32(W_up[k * N + col]);
            }
        }
    }

    shared_gate[tid] = gate_partial;
    shared_up[tid] = up_partial;
    workgroupBarrier();

    var stride: u32 = threads_per_col >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (thread_in_col < stride) {
            let lane = col_within_tile * threads_per_col + thread_in_col;
            shared_gate[lane] = shared_gate[lane] + shared_gate[lane + stride];
            shared_up[lane] = shared_up[lane] + shared_up[lane + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (thread_in_col == 0u && col_valid) {
        let lane = col_within_tile * threads_per_col;
        let gate_val = shared_gate[lane];
        let up_val = shared_up[lane];
        output[row * N + col] = f16(gelu_tanh(gate_val) * up_val);
    }
}
