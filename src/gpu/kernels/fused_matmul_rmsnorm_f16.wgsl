// Fused GEMV + RMSNorm Kernel (F16 variant)
//
// For decode (M=1), combines the down projection matmul with RMSNorm in a single kernel:
// 1. Compute GEMV: C[1, N] = A[1, K] × B[K, N]  (down projection)
// 2. Compute RMSNorm on C: output = C / sqrt(mean(C^2) + eps) * weight
// 3. Optional residual: output = output + residual
//
// F16 variant: All buffers are f16, accumulation in f32 for stability.
//
// Benefits:
// - Single GPU dispatch instead of 2
// - No intermediate buffer for matmul output
// - Better cache locality
// - 2x memory bandwidth vs F32 variant
//
// Expected speedup: 1.3-1.8x for post-FFN normalization path

enable f16;

override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;
override COLS_PER_WG: u32 = 4u;
override THREADS_PER_COL: u32 = 64u;  // WORKGROUP_SIZE / COLS_PER_WG
override RMS_NORM_OFFSET: bool = false;
override WEIGHT_IS_F16: bool = false;

struct Uniforms {
    N: u32,             // Output dimension (hidden_size)
    K: u32,             // Input dimension (intermediate_size)
    eps: f32,           // RMSNorm epsilon
    has_residual: u32,  // 1 if residual provided, 0 otherwise
    transpose_b: u32,   // 1 if weight is [N,K] (row-major), 0 if [K,N] (column-major)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;       // [1, K] - activation from FFN
@group(0) @binding(2) var<storage, read> weight: array<f16>;      // [K, N] - down projection weight
@group(0) @binding(3) var<storage, read> norm_weight: array<u32>; // [N] - RMSNorm weight (F16 or F32)
@group(0) @binding(4) var<storage, read_write> output: array<f16>;// [1, N] - final output
@group(0) @binding(5) var<storage, read> residual: array<f16>;    // [1, N] - optional residual

// Shared memory for reduction (f32 for numerical stability)
var<workgroup> shared_partial: array<f32, MAX_WORKGROUP_SIZE>;  // Partial dot products
var<workgroup> shared_output: array<f32, MAX_WORKGROUP_SIZE>;   // Output values for RMSNorm
var<workgroup> shared_sum_sq: array<f32, MAX_WORKGROUP_SIZE>;   // Sum of squares for reduction

fn apply_weight(w: f32) -> f32 {
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

// Optimized single-workgroup variant for small N (hidden_size <= WORKGROUP_SIZE)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv_rmsnorm_small(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let N = u.N;
    let K = u.K;

    // Each thread computes one output column (for N <= WORKGROUP_SIZE)
    var dot_sum: f32 = 0.0;
    if (tid < N) {
        // Unroll by 4 for better ILP
        var k: u32 = 0u;
        let k_aligned = (K / 4u) * 4u;

        for (; k < k_aligned; k = k + 4u) {
            let a0 = f32(input[k]);
            let a1 = f32(input[k + 1u]);
            let a2 = f32(input[k + 2u]);
            let a3 = f32(input[k + 3u]);

            if (u.transpose_b == 1u) {
                // Row-major: weight[tid, k]
                let base = tid * K + k;
                dot_sum = dot_sum + a0 * f32(weight[base])
                                  + a1 * f32(weight[base + 1u])
                                  + a2 * f32(weight[base + 2u])
                                  + a3 * f32(weight[base + 3u]);
            } else {
                // Column-major: weight[k, tid]
                dot_sum = dot_sum + a0 * f32(weight[k * N + tid])
                                  + a1 * f32(weight[(k + 1u) * N + tid])
                                  + a2 * f32(weight[(k + 2u) * N + tid])
                                  + a3 * f32(weight[(k + 3u) * N + tid]);
            }
        }

        // Handle remainder
        for (; k < K; k = k + 1u) {
            let w_idx = select(k * N + tid, tid * K + k, u.transpose_b == 1u);
            dot_sum = dot_sum + f32(input[k]) * f32(weight[w_idx]);
        }
    }

    // Store output value and compute sum of squares
    shared_output[tid] = dot_sum;
    shared_sum_sq[tid] = select(0.0, dot_sum * dot_sum, tid < N);
    workgroupBarrier();

    // Parallel reduction for sum of squares
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_sum_sq[tid] = shared_sum_sq[tid] + shared_sum_sq[tid + stride];
        }
        workgroupBarrier();
    }

    // Compute inverse RMS
    let mean_sq = shared_sum_sq[0] / f32(N);
    let inv_rms = 1.0 / sqrt(mean_sq + u.eps);

    // Write normalized output
    if (tid < N) {
        var result = shared_output[tid] * inv_rms * apply_weight(load_norm_weight(tid));
        if (u.has_residual == 1u) {
            result = result + f32(residual[tid]);
        }
        output[tid] = f16(result);
    }
}

// Medium variant for N up to ~4096 (single workgroup, multiple elements per thread)
// Handles Gemma's hidden_size=2304 and similar models
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv_rmsnorm_medium(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let N = u.N;
    let K = u.K;

    // Each thread handles multiple output columns
    let cols_per_thread = (N + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // Phase 1: Compute dot products for all columns assigned to this thread
    var local_sum_sq: f32 = 0.0;

    for (var c: u32 = 0u; c < cols_per_thread; c = c + 1u) {
        let col = tid + c * WORKGROUP_SIZE;
        if (col < N) {
            // Compute dot product for this column
            var dot_sum: f32 = 0.0;

            // Unroll by 4 for better ILP
            var k: u32 = 0u;
            let k_aligned = (K / 4u) * 4u;

            for (; k < k_aligned; k = k + 4u) {
                let a0 = f32(input[k]);
                let a1 = f32(input[k + 1u]);
                let a2 = f32(input[k + 2u]);
                let a3 = f32(input[k + 3u]);

                if (u.transpose_b == 1u) {
                    let base = col * K + k;
                    dot_sum = dot_sum + a0 * f32(weight[base])
                                      + a1 * f32(weight[base + 1u])
                                      + a2 * f32(weight[base + 2u])
                                      + a3 * f32(weight[base + 3u]);
                } else {
                    dot_sum = dot_sum + a0 * f32(weight[k * N + col])
                                      + a1 * f32(weight[(k + 1u) * N + col])
                                      + a2 * f32(weight[(k + 2u) * N + col])
                                      + a3 * f32(weight[(k + 3u) * N + col]);
                }
            }

            // Handle remainder
            for (; k < K; k = k + 1u) {
                let w_idx = select(k * N + col, col * K + k, u.transpose_b == 1u);
                dot_sum = dot_sum + f32(input[k]) * f32(weight[w_idx]);
            }

            // Store to output buffer (will normalize in-place later)
            output[col] = f16(dot_sum);
            local_sum_sq = local_sum_sq + dot_sum * dot_sum;
        }
    }

    // Phase 2: Parallel reduction for sum of squares
    shared_sum_sq[tid] = local_sum_sq;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_sum_sq[tid] = shared_sum_sq[tid] + shared_sum_sq[tid + stride];
        }
        workgroupBarrier();
    }

    // Compute inverse RMS (all threads read the same value)
    let mean_sq = shared_sum_sq[0] / f32(N);
    let inv_rms = 1.0 / sqrt(mean_sq + u.eps);

    // Phase 3: Normalize outputs in-place
    for (var c: u32 = 0u; c < cols_per_thread; c = c + 1u) {
        let col = tid + c * WORKGROUP_SIZE;
        if (col < N) {
            var result = f32(output[col]) * inv_rms * apply_weight(load_norm_weight(col));
            if (u.has_residual == 1u) {
                result = result + f32(residual[col]);
            }
            output[col] = f16(result);
        }
    }
}

// Main entry point - dispatches based on N size
// Uses multiple workgroups for large N
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wgs: vec3<u32>
) {
    let tid = local_id.x;
    let N = u.N;
    let K = u.K;

    // Each workgroup handles COLS_PER_WG columns of output
    let col_in_wg = tid / THREADS_PER_COL;
    let thread_in_col = tid % THREADS_PER_COL;
    let global_col = wg_id.x * COLS_PER_WG + col_in_wg;

    // Phase 1: Compute dot product for this column
    var dot_sum: f32 = 0.0;
    if (global_col < N) {
        let k_per_thread = (K + THREADS_PER_COL - 1u) / THREADS_PER_COL;
        let k_start = thread_in_col * k_per_thread;
        let k_end = min(k_start + k_per_thread, K);

        for (var k: u32 = k_start; k < k_end; k = k + 1u) {
            let w_idx = select(k * N + global_col, global_col * K + k, u.transpose_b == 1u);
            dot_sum = dot_sum + f32(input[k]) * f32(weight[w_idx]);
        }
    }

    // Store partial dot product in shared memory
    shared_partial[tid] = dot_sum;
    workgroupBarrier();

    // Reduce within each column
    for (var stride: u32 = THREADS_PER_COL / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_in_col < stride) {
            let idx = col_in_wg * THREADS_PER_COL + thread_in_col;
            shared_partial[idx] = shared_partial[idx] + shared_partial[idx + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 of each column has the final dot product
    var matmul_output: f32 = 0.0;
    if (thread_in_col == 0u && global_col < N) {
        matmul_output = shared_partial[col_in_wg * THREADS_PER_COL];
        shared_output[col_in_wg] = matmul_output;
        shared_sum_sq[col_in_wg] = matmul_output * matmul_output;
    }
    workgroupBarrier();

    // Reduce sum of squares across columns in this workgroup
    for (var stride: u32 = COLS_PER_WG / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride && tid + stride < COLS_PER_WG) {
            shared_sum_sq[tid] = shared_sum_sq[tid] + shared_sum_sq[tid + stride];
        }
        workgroupBarrier();
    }

    let local_sum_sq = shared_sum_sq[0];

    // Phase 3: Write normalized output
    if (thread_in_col == 0u && global_col < N) {
        let val = shared_output[col_in_wg];
        let mean_sq = local_sum_sq / f32(min(COLS_PER_WG, N));
        let inv_rms = 1.0 / sqrt(mean_sq + u.eps);

        var result = val * inv_rms * apply_weight(load_norm_weight(global_col));

        if (u.has_residual == 1u) {
            result = result + f32(residual[global_col]);
        }

        output[global_col] = f16(result);
    }
}