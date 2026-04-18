// Fused RMSNorm + Matmul Kernel (tiled, F16)
//
// Fuses a pre-projection RMSNorm with the subsequent matmul into a single
// compute pass: `output[m, n] = sum_k( rmsnorm(input[m, :])[k] * weight[k, n] )`.
// Applies to the standard transformer prefill boundaries:
//   - input_norm → q_proj / k_proj / v_proj
//   - pre_ffn_norm → gate_proj / up_proj
//
// Each workgroup owns one query row m AND one column tile of the output. The
// row normalisation (sum-of-squares + scale) is computed cooperatively by the
// workgroup before any thread consumes the normed row for its column tile, so
// the row is materialised once per workgroup into shared memory.
//
// Dispatch: (ceil(N / COLS_PER_WG), M, 1)
//
// Correctness notes:
//   - Accumulation in f32 for numerical stability.
//   - `u.transpose_b` toggles between B=[K,N] (row-major) and B=[N,K] layouts.
//   - `RMS_NORM_OFFSET` matches Gemma-family manifests where the stored weight
//     encodes `(weight - 1.0)` rather than the literal scale.
//   - `WEIGHT_IS_F16` toggles between f16-packed (2 values per u32) and f32
//     norm-weight storage.
//   - No thread returns on a divergent condition mid-kernel: every barrier is
//     reached by every thread in the workgroup. Divergent column-in-tile
//     indices either compute zero or skip the final write only.

enable f16;

override WORKGROUP_SIZE: u32 = 128u;
override COLS_PER_WG: u32 = 8u;
override RMS_NORM_OFFSET: bool = false;
override WEIGHT_IS_F16: bool = false;

const MAX_K: u32 = 2048u;
const MAX_WG: u32 = 256u;

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    eps: f32,
    transpose_b: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;        // [M, K]
@group(0) @binding(2) var<storage, read> matmul_weight: array<f16>;// [K, N] or [N, K]
@group(0) @binding(3) var<storage, read> norm_weight: array<u32>;  // [K] f16 packed or f32
@group(0) @binding(4) var<storage, read_write> output: array<f16>; // [M, N]

var<workgroup> shared_sum_sq: array<f32, MAX_WG>;
var<workgroup> shared_normed: array<f32, MAX_K>;
var<workgroup> shared_scale: f32;
var<workgroup> shared_dot: array<f32, MAX_WG>;

fn apply_norm_weight(w: f32) -> f32 {
    if (RMS_NORM_OFFSET) {
        return 1.0 + w;
    }
    return w;
}

fn load_norm_weight(idx: u32) -> f32 {
    if (WEIGHT_IS_F16) {
        let packed = norm_weight[idx >> 1u];
        let pair = unpack2x16float(packed);
        return select(pair.x, pair.y, (idx & 1u) == 1u);
    }
    return bitcast<f32>(norm_weight[idx]);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let col_tile = wg.x;
    let row = wg.y;
    let tid = lid.x;

    // Every thread in the workgroup has the same row — either all return or
    // all participate.
    if (row >= u.M) {
        return;
    }

    let N = u.N;
    let K = u.K;
    let row_base = row * K;

    // -------------------------------------------------------------------------
    // Stage 1 — cooperative sum-of-squares over the input row.
    // -------------------------------------------------------------------------
    var local_ss: f32 = 0.0;
    for (var k: u32 = tid; k < K; k = k + WORKGROUP_SIZE) {
        let x = f32(input[row_base + k]);
        local_ss = local_ss + x * x;
    }
    shared_sum_sq[tid] = local_ss;
    workgroupBarrier();

    var stride: u32 = WORKGROUP_SIZE >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_sum_sq[tid] = shared_sum_sq[tid] + shared_sum_sq[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (tid == 0u) {
        let mean_sq = shared_sum_sq[0] / f32(K);
        shared_scale = 1.0 / sqrt(mean_sq + u.eps);
    }
    workgroupBarrier();
    let scale = shared_scale;

    // -------------------------------------------------------------------------
    // Stage 2 — materialise the normed row once into shared memory.
    // -------------------------------------------------------------------------
    for (var k: u32 = tid; k < K; k = k + WORKGROUP_SIZE) {
        let x = f32(input[row_base + k]);
        let w = apply_norm_weight(load_norm_weight(k));
        shared_normed[k] = x * scale * w;
    }
    workgroupBarrier();

    // -------------------------------------------------------------------------
    // Stage 3 — column-tile matmul. Each workgroup computes COLS_PER_WG
    // adjacent output columns; threads_per_col threads cooperate on each one.
    // Threads whose col is out of range compute zero and skip only the write.
    // -------------------------------------------------------------------------
    let threads_per_col = WORKGROUP_SIZE / COLS_PER_WG;
    let col_within_tile = tid / threads_per_col;
    let thread_in_col = tid % threads_per_col;
    let col = col_tile * COLS_PER_WG + col_within_tile;
    let col_valid = col < N;

    var dot_partial: f32 = 0.0;
    if (col_valid) {
        if (u.transpose_b != 0u) {
            let w_base = col * K;
            for (var k: u32 = thread_in_col; k < K; k = k + threads_per_col) {
                dot_partial = dot_partial + shared_normed[k] * f32(matmul_weight[w_base + k]);
            }
        } else {
            for (var k: u32 = thread_in_col; k < K; k = k + threads_per_col) {
                dot_partial = dot_partial + shared_normed[k] * f32(matmul_weight[k * N + col]);
            }
        }
    }

    shared_dot[tid] = dot_partial;
    workgroupBarrier();

    var reduce_stride: u32 = threads_per_col >> 1u;
    loop {
        if (reduce_stride == 0u) { break; }
        if (thread_in_col < reduce_stride) {
            let lane = col_within_tile * threads_per_col + thread_in_col;
            shared_dot[lane] = shared_dot[lane] + shared_dot[lane + reduce_stride];
        }
        workgroupBarrier();
        reduce_stride = reduce_stride >> 1u;
    }

    if (thread_in_col == 0u && col_valid) {
        let lane = col_within_tile * threads_per_col;
        output[row * N + col] = f16(shared_dot[lane]);
    }
}
