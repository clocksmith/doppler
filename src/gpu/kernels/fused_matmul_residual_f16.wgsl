// Fused Matmul + Residual Kernel (F16)
//
// Fuses a projection matmul with the residual add that immediately follows it:
//   output[m, n] = sum_k(input[m, k] * weight[k, n]) + residual[m, n]
//
// Applies to the standard transformer prefill boundaries:
//   - o_proj → attn_residual
//   - down_proj → ffn_residual
//
// Each workgroup owns one row m and a column tile. Weight reads are coalesced
// across threads within the column tile. Eliminating the separate residual
// pass saves the full CPU submit + GPU schedule overhead of the standalone
// residual dispatch.
//
// Dispatch: (ceil(N / COLS_PER_WG), M, 1)
//
// Accumulation in f32 for numerical stability; output is f16 to match the
// existing residual-stream storage contract.
//
// No thread returns on a divergent condition mid-kernel: every barrier is
// reached by every thread in the workgroup.

enable f16;

override WORKGROUP_SIZE: u32 = 128u;
override COLS_PER_WG: u32 = 8u;

const MAX_WG: u32 = 256u;

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    transpose_b: u32,  // 1 if weight is [N, K], 0 if [K, N]
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;       // [M, K]
@group(0) @binding(2) var<storage, read> weight: array<f16>;      // [K, N] or [N, K]
@group(0) @binding(3) var<storage, read> residual: array<f16>;    // [M, N]
@group(0) @binding(4) var<storage, read_write> output: array<f16>;// [M, N]

var<workgroup> shared_dot: array<f32, MAX_WG>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let col_tile = wg.x;
    let row = wg.y;
    let tid = lid.x;
    if (row >= u.M) { return; }

    let N = u.N;
    let K = u.K;

    let threads_per_col = WORKGROUP_SIZE / COLS_PER_WG;
    let col_within_tile = tid / threads_per_col;
    let thread_in_col = tid % threads_per_col;
    let col = col_tile * COLS_PER_WG + col_within_tile;
    let col_valid = col < N;

    var dot_partial: f32 = 0.0;
    if (col_valid) {
        let input_row_base = row * K;
        if (u.transpose_b != 0u) {
            let w_base = col * K;
            for (var k: u32 = thread_in_col; k < K; k = k + threads_per_col) {
                dot_partial = dot_partial
                    + f32(input[input_row_base + k]) * f32(weight[w_base + k]);
            }
        } else {
            for (var k: u32 = thread_in_col; k < K; k = k + threads_per_col) {
                dot_partial = dot_partial
                    + f32(input[input_row_base + k]) * f32(weight[k * N + col]);
            }
        }
    }

    shared_dot[tid] = dot_partial;
    workgroupBarrier();

    var stride: u32 = threads_per_col >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (thread_in_col < stride) {
            let lane = col_within_tile * threads_per_col + thread_in_col;
            shared_dot[lane] = shared_dot[lane] + shared_dot[lane + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (thread_in_col == 0u && col_valid) {
        let lane = col_within_tile * threads_per_col;
        let sum = shared_dot[lane];
        let resid = f32(residual[row * N + col]);
        output[row * N + col] = f16(sum + resid);
    }
}
