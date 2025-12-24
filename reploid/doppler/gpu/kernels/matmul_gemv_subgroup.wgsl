// Subgroup-Optimized GEMV Kernel
// For M=1 decode: C[N] = A[K] * B[K,N] or A[K] * B^T[N,K]
//
// Key optimizations over base GEMV:
// 1. Use subgroupAdd() for reduction - much faster than shared memory
// 2. Vectorized vec4 loads for weights
// 3. Each workgroup processes multiple output columns
// 4. Loop unrolling for better ILP
//
// A is f32 (activations), B is f16 (weights), C is f32.
// transposeB=0: B is [K, N] (GGUF/column-major), access B[k * N + col]
// transposeB=1: B is [N, K] (SafeTensors/row-major), access B[col * K + k]
//
// IMPORTANT: This kernel maintains uniform control flow for subgroup operations.
// All threads execute subgroupAdd - invalid threads contribute 0.

enable f16;
enable subgroups;

const WG_SIZE: u32 = 256u;
const COLS_PER_WG: u32 = 4u;  // Each workgroup computes 4 output columns
const THREADS_PER_COL: u32 = 64u;  // 256 / 4 = 64 threads per column
const MAX_SUBGROUPS_PER_COL: u32 = 16u;  // Support sg_size >= 4 (64/4 = 16)

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    transposeB: u32,
    workgroups_x: u32,  // For 2D dispatch when N > 65535*4
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f16>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

// Shared memory for final reduction across subgroups
// Size: 16 subgroups * 4 columns = 64 (supports sg_size >= 4)
var<workgroup> wg_sums: array<f32, 64>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32
) {
    let local_id = lid.x;

    // Which column within this workgroup's set of COLS_PER_WG
    let col_in_wg = local_id / THREADS_PER_COL;
    let thread_in_col = local_id % THREADS_PER_COL;

    // Global output column (supports 2D dispatch for large N)
    // Linear workgroup ID = wg_id.y * workgroups_x + wg_id.x
    let wg_linear = wg_id.y * uniforms.workgroups_x + wg_id.x;
    let base_col = wg_linear * COLS_PER_WG;
    let col = base_col + col_in_wg;

    // Track validity - NO early return to maintain uniform control flow
    let is_valid = col < uniforms.N;

    // Each thread computes partial sum for its assigned k values
    var partial_sum: f32 = 0.0;

    // Only do work if this column is valid
    if (is_valid) {
        // Process K in chunks, each thread handles K/64 elements
        let k_per_thread = (uniforms.K + THREADS_PER_COL - 1u) / THREADS_PER_COL;
        let k_start = thread_in_col * k_per_thread;
        let k_end = min(k_start + k_per_thread, uniforms.K);

        // Main loop - process 4 elements at a time when aligned
        var k = k_start;
        let k_aligned_end = k_start + ((k_end - k_start) / 4u) * 4u;

        if (uniforms.transposeB == 1u) {
            // B is [N, K] (SafeTensors/row-major): B[col, k] = B[col * K + k]
            let b_row_offset = col * uniforms.K;

            for (; k < k_aligned_end; k = k + 4u) {
                let a0 = A[k];
                let a1 = A[k + 1u];
                let a2 = A[k + 2u];
                let a3 = A[k + 3u];

                let b0 = f32(B[b_row_offset + k]);
                let b1 = f32(B[b_row_offset + k + 1u]);
                let b2 = f32(B[b_row_offset + k + 2u]);
                let b3 = f32(B[b_row_offset + k + 3u]);

                partial_sum = partial_sum + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            }

            for (; k < k_end; k = k + 1u) {
                partial_sum = partial_sum + A[k] * f32(B[b_row_offset + k]);
            }
        } else {
            // B is [K, N] (GGUF/column-major): B[k, col] = B[k * N + col]
            for (; k < k_aligned_end; k = k + 4u) {
                let a0 = A[k];
                let a1 = A[k + 1u];
                let a2 = A[k + 2u];
                let a3 = A[k + 3u];

                let b0 = f32(B[k * uniforms.N + col]);
                let b1 = f32(B[(k + 1u) * uniforms.N + col]);
                let b2 = f32(B[(k + 2u) * uniforms.N + col]);
                let b3 = f32(B[(k + 3u) * uniforms.N + col]);

                partial_sum = partial_sum + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            }

            for (; k < k_end; k = k + 1u) {
                partial_sum = partial_sum + A[k] * f32(B[k * uniforms.N + col]);
            }
        }
    }

    // Subgroup reduction - ALL threads must execute this (uniform control flow)
    // Invalid threads contribute 0 to the sum
    let sg_sum = subgroupAdd(partial_sum);

    // Only one thread per subgroup writes to shared memory
    let num_subgroups_per_col = (THREADS_PER_COL + sg_size - 1u) / sg_size;

    if (sg_id == 0u && thread_in_col < THREADS_PER_COL) {
        let sg_idx_in_col = thread_in_col / sg_size;
        wg_sums[col_in_wg * MAX_SUBGROUPS_PER_COL + sg_idx_in_col] = sg_sum;
    }

    workgroupBarrier();

    // Final reduction - first thread of each column sums subgroup results
    if (thread_in_col == 0u && is_valid) {
        var final_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < num_subgroups_per_col; i = i + 1u) {
            final_sum = final_sum + wg_sums[col_in_wg * MAX_SUBGROUPS_PER_COL + i];
        }
        C[col] = final_sum * uniforms.alpha;
    }
}

// ============================================================================
// Multi-column GEMV for large vocab (LM head)
// ============================================================================
// For very large N (vocab=262144), 4 cols/workgroup still means 65K workgroups.
// This variant processes 32 columns per workgroup:
// - 262144/32 = 8192 workgroups (8x fewer than base kernel)
// - Each thread handles more K elements, better amortizing A loads
//
// Layout: 256 threads = 8 threads per column × 32 columns
const MULTICOL_COLS_PER_WG: u32 = 32u;
const MULTICOL_THREADS_PER_COL: u32 = 8u;  // 256 / 32 = 8
const MULTICOL_MAX_SUBGROUPS: u32 = 8u;    // Support sg_size >= 1 (unlikely but safe)

// Shared memory: 32 columns × 8 values = 256
var<workgroup> multicol_wg_sums: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main_multicol(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32
) {
    let local_id = lid.x;

    // Which column within this workgroup (0..31)
    let col_in_wg = local_id / MULTICOL_THREADS_PER_COL;
    // Which thread within the column (0..7)
    let thread_in_col = local_id % MULTICOL_THREADS_PER_COL;

    // Global output column (supports 2D dispatch)
    let wg_linear = wg_id.y * uniforms.workgroups_x + wg_id.x;
    let base_col = wg_linear * MULTICOL_COLS_PER_WG;
    let col = base_col + col_in_wg;

    // Track validity
    let is_valid = col < uniforms.N;

    var partial_sum: f32 = 0.0;

    if (is_valid) {
        // Each of 8 threads splits K
        let k_per_thread = (uniforms.K + MULTICOL_THREADS_PER_COL - 1u) / MULTICOL_THREADS_PER_COL;
        let k_start = thread_in_col * k_per_thread;
        let k_end = min(k_start + k_per_thread, uniforms.K);

        // Unroll by 4 for ILP
        var k = k_start;
        let k_aligned_end = k_start + ((k_end - k_start) / 4u) * 4u;

        if (uniforms.transposeB == 1u) {
            // B is [N, K] (SafeTensors/row-major): B[col, k] = B[col * K + k]
            let b_row_offset = col * uniforms.K;

            for (; k < k_aligned_end; k = k + 4u) {
                let a0 = A[k];
                let a1 = A[k + 1u];
                let a2 = A[k + 2u];
                let a3 = A[k + 3u];

                let b0 = f32(B[b_row_offset + k]);
                let b1 = f32(B[b_row_offset + k + 1u]);
                let b2 = f32(B[b_row_offset + k + 2u]);
                let b3 = f32(B[b_row_offset + k + 3u]);

                partial_sum = partial_sum + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            }

            for (; k < k_end; k = k + 1u) {
                partial_sum = partial_sum + A[k] * f32(B[b_row_offset + k]);
            }
        } else {
            // B is [K, N] (GGUF/column-major): B[k, col] = B[k * N + col]
            for (; k < k_aligned_end; k = k + 4u) {
                let a0 = A[k];
                let a1 = A[k + 1u];
                let a2 = A[k + 2u];
                let a3 = A[k + 3u];

                let b0 = f32(B[k * uniforms.N + col]);
                let b1 = f32(B[(k + 1u) * uniforms.N + col]);
                let b2 = f32(B[(k + 2u) * uniforms.N + col]);
                let b3 = f32(B[(k + 3u) * uniforms.N + col]);

                partial_sum = partial_sum + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            }

            for (; k < k_end; k = k + 1u) {
                partial_sum = partial_sum + A[k] * f32(B[k * uniforms.N + col]);
            }
        }
    }

    // Write partial sums to shared memory for reduction
    multicol_wg_sums[local_id] = partial_sum;
    workgroupBarrier();

    // Thread 0 of each column reduces its 8 values
    if (thread_in_col == 0u && is_valid) {
        var final_sum: f32 = 0.0;
        let base = col_in_wg * MULTICOL_THREADS_PER_COL;
        for (var i: u32 = 0u; i < MULTICOL_THREADS_PER_COL; i = i + 1u) {
            final_sum = final_sum + multicol_wg_sums[base + i];
        }
        C[col] = final_sum * uniforms.alpha;
    }
}

// Alternative entry point with vec4 weight loads (requires K % 4 == 0)
@compute @workgroup_size(256, 1, 1)
fn main_vec4(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32
) {
    let local_id = lid.x;
    let col_in_wg = local_id / THREADS_PER_COL;
    let thread_in_col = local_id % THREADS_PER_COL;

    // Global output column (supports 2D dispatch for large N)
    let wg_linear = wg_id.y * uniforms.workgroups_x + wg_id.x;
    let base_col = wg_linear * COLS_PER_WG;
    let col = base_col + col_in_wg;

    // Track validity - NO early return to maintain uniform control flow
    let is_valid = col < uniforms.N;

    var partial_sum: f32 = 0.0;

    if (is_valid) {
        // K is guaranteed to be multiple of 4
        let K4 = uniforms.K / 4u;
        let k4_per_thread = (K4 + THREADS_PER_COL - 1u) / THREADS_PER_COL;
        let k4_start = thread_in_col * k4_per_thread;
        let k4_end = min(k4_start + k4_per_thread, K4);

        if (uniforms.transposeB == 1u) {
            // B is [N, K] (SafeTensors/row-major): B[col, k] = B[col * K + k]
            let b_row_offset = col * uniforms.K;

            for (var k4: u32 = k4_start; k4 < k4_end; k4 = k4 + 1u) {
                let k = k4 * 4u;

                let a = vec4<f32>(A[k], A[k + 1u], A[k + 2u], A[k + 3u]);

                let b = vec4<f32>(
                    f32(B[b_row_offset + k]),
                    f32(B[b_row_offset + k + 1u]),
                    f32(B[b_row_offset + k + 2u]),
                    f32(B[b_row_offset + k + 3u])
                );

                partial_sum = partial_sum + dot(a, b);
            }
        } else {
            // B is [K, N] (GGUF/column-major): B[k, col] = B[k * N + col]
            for (var k4: u32 = k4_start; k4 < k4_end; k4 = k4 + 1u) {
                let k = k4 * 4u;

                let a = vec4<f32>(A[k], A[k + 1u], A[k + 2u], A[k + 3u]);

                let b = vec4<f32>(
                    f32(B[k * uniforms.N + col]),
                    f32(B[(k + 1u) * uniforms.N + col]),
                    f32(B[(k + 2u) * uniforms.N + col]),
                    f32(B[(k + 3u) * uniforms.N + col])
                );

                partial_sum = partial_sum + dot(a, b);
            }
        }
    }

    // Subgroup reduction - ALL threads must execute this (uniform control flow)
    let sg_sum = subgroupAdd(partial_sum);

    let num_subgroups_per_col = (THREADS_PER_COL + sg_size - 1u) / sg_size;

    if (sg_id == 0u && thread_in_col < THREADS_PER_COL) {
        let sg_idx_in_col = thread_in_col / sg_size;
        wg_sums[col_in_wg * MAX_SUBGROUPS_PER_COL + sg_idx_in_col] = sg_sum;
    }

    workgroupBarrier();

    if (thread_in_col == 0u && is_valid) {
        var final_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < num_subgroups_per_col; i = i + 1u) {
            final_sum = final_sum + wg_sums[col_in_wg * MAX_SUBGROUPS_PER_COL + i];
        }
        C[col] = final_sum * uniforms.alpha;
    }
}
