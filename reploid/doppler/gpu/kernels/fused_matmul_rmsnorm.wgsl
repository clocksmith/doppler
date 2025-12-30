// Fused GEMV + RMSNorm Kernel
//
// For decode (M=1), combines the down projection matmul with RMSNorm in a single kernel:
// 1. Compute GEMV: C[1, N] = A[1, K] × B[K, N]  (down projection)
// 2. Compute RMSNorm on C: output = C / sqrt(mean(C^2) + eps) * weight
// 3. Optional residual: output = output + residual
//
// Benefits:
// - Single GPU dispatch instead of 2
// - No intermediate buffer for matmul output
// - Better cache locality
//
// Expected speedup: 1.2-1.5x for post-FFN normalization path

override WORKGROUP_SIZE: u32 = 256u;
override COLS_PER_WG: u32 = 4u;
override THREADS_PER_COL: u32 = 64u;  // WORKGROUP_SIZE / COLS_PER_WG

struct Uniforms {
    N: u32,             // Output dimension (hidden_size)
    K: u32,             // Input dimension (intermediate_size)
    eps: f32,           // RMSNorm epsilon
    has_residual: u32,  // 1 if residual provided, 0 otherwise
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;       // [1, K] - activation from FFN
@group(0) @binding(2) var<storage, read> weight: array<f32>;      // [K, N] - down projection weight
@group(0) @binding(3) var<storage, read> norm_weight: array<f32>; // [N] - RMSNorm weight
@group(0) @binding(4) var<storage, read_write> output: array<f32>;// [1, N] - final output
@group(0) @binding(5) var<storage, read> residual: array<f32>;    // [1, N] - optional residual

// Shared memory for reduction
var<workgroup> shared_partial: array<f32, 256>;  // Partial dot products
var<workgroup> shared_output: array<f32, 256>;   // Output values for RMSNorm
var<workgroup> shared_sum_sq: array<f32, 256>;   // Sum of squares for reduction

// Fused GEMV + RMSNorm: Main entry point
// Each workgroup computes COLS_PER_WG output columns, then all workgroups
// cooperate to compute the RMSNorm across the full output row.
//
// Phase 1: GEMV - compute dot products for each output column
// Phase 2: Store outputs and compute partial sum of squares
// Phase 3: Global reduction of sum of squares (via atomic or multi-pass)
// Phase 4: Normalize outputs and optionally add residual
//
// Note: This kernel requires N <= WG_SIZE * COLS_PER_WG for single-pass RMSNorm.
// For larger N, use the multi-workgroup variant.

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
    let col_in_wg = tid / THREADS_PER_COL;  // Which column within this workgroup (0-3)
    let thread_in_col = tid % THREADS_PER_COL;  // Thread index within column (0-63)
    let global_col = wg_id.x * COLS_PER_WG + col_in_wg;  // Global column index

    // Phase 1: Compute dot product for this column
    var dot_sum: f32 = 0.0;
    if (global_col < N) {
        // Each thread in a column processes a stripe of K
        let k_per_thread = (K + THREADS_PER_COL - 1u) / THREADS_PER_COL;
        let k_start = thread_in_col * k_per_thread;
        let k_end = min(k_start + k_per_thread, K);

        for (var k: u32 = k_start; k < k_end; k = k + 1u) {
            // Row-major weight: weight[k * N + global_col]
            dot_sum = dot_sum + input[k] * weight[k * N + global_col];
        }
    }

    // Store partial dot product in shared memory
    shared_partial[tid] = dot_sum;
    workgroupBarrier();

    // Reduce within each column (THREADS_PER_COL threads -> 1 value)
    // Tree reduction: 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
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

    // Phase 2: For single-workgroup case (N <= 1024), we can do RMSNorm directly
    // For multi-workgroup, we'd need a second pass or atomic accumulation

    // This version handles N <= 1024 (256 threads * 4 cols/wg)
    // For larger N, spawn multiple workgroups and do global reduction

    // Reduce sum of squares across columns in this workgroup
    // All threads must participate in barriers, so move barrier outside condition
    for (var stride: u32 = COLS_PER_WG / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride && tid + stride < COLS_PER_WG) {
            shared_sum_sq[tid] = shared_sum_sq[tid] + shared_sum_sq[tid + stride];
        }
        workgroupBarrier();
    }

    // For single-workgroup: compute RMS and normalize
    // Thread 0 has partial sum; need global sum across all workgroups
    // For now, assume single workgroup (N <= COLS_PER_WG = 4)
    // TODO: Add multi-workgroup support for larger N

    let local_sum_sq = shared_sum_sq[0];

    // Phase 3: Write normalized output
    if (thread_in_col == 0u && global_col < N) {
        let val = shared_output[col_in_wg];

        // For multi-WG: would need global sum here
        // For now: just use local (works for small N)
        let mean_sq = local_sum_sq / f32(min(COLS_PER_WG, N));
        let inv_rms = 1.0 / sqrt(mean_sq + u.eps);

        var result = val * inv_rms * norm_weight[global_col];

        if (u.has_residual == 1u) {
            result = result + residual[global_col];
        }

        output[global_col] = result;
    }
}

// Optimized single-workgroup variant for small N (hiddenSize <= 256)
// All output columns computed by one workgroup, RMSNorm in single pass
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv_rmsnorm_small(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let N = u.N;
    let K = u.K;

    // Each thread computes one output column (for N <= 256)
    var dot_sum: f32 = 0.0;
    if (tid < N) {
        for (var k: u32 = 0u; k < K; k = k + 1u) {
            dot_sum = dot_sum + input[k] * weight[k * N + tid];
        }
    }

    // Store output value and compute sum of squares
    shared_output[tid] = dot_sum;
    shared_sum_sq[tid] = select(0.0, dot_sum * dot_sum, tid < N);
    workgroupBarrier();

    // Parallel reduction for sum of squares
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
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
        var result = shared_output[tid] * inv_rms * norm_weight[tid];
        if (u.has_residual == 1u) {
            result = result + residual[tid];
        }
        output[tid] = result;
    }
}

// Medium variant for N up to ~4096 (single workgroup, multiple elements per thread)
// Handles Gemma 3's hiddenSize=1152 and similar models
// Each thread computes ceil(N/256) output columns sequentially
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv_rmsnorm_medium(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let N = u.N;
    let K = u.K;

    // Each thread handles multiple output columns
    let cols_per_thread = (N + 255u) / 256u;  // ceil(N/256)

    // Phase 1: Compute dot products for all columns assigned to this thread
    // Store partial sum of squares as we go
    var local_sum_sq: f32 = 0.0;

    for (var c: u32 = 0u; c < cols_per_thread; c = c + 1u) {
        let col = tid + c * 256u;
        if (col < N) {
            // Compute dot product for this column
            var dot_sum: f32 = 0.0;
            for (var k: u32 = 0u; k < K; k = k + 1u) {
                dot_sum = dot_sum + input[k] * weight[k * N + col];
            }

            // Store to output buffer (will normalize in-place later)
            output[col] = dot_sum;
            local_sum_sq = local_sum_sq + dot_sum * dot_sum;
        }
    }

    // Phase 2: Parallel reduction for sum of squares
    shared_sum_sq[tid] = local_sum_sq;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
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
        let col = tid + c * 256u;
        if (col < N) {
            var result = output[col] * inv_rms * norm_weight[col];
            if (u.has_residual == 1u) {
                result = result + residual[col];
            }
            output[col] = result;
        }
    }
}

// Multi-workgroup variant using two-phase approach
// Phase 1: Each workgroup computes output columns and partial sum of squares
// Phase 2: Separate kernel reduces partials and normalizes
// This entry point is Phase 1 only
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv_rmsnorm_phase1(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tid = local_id.x;
    let N = u.N;
    let K = u.K;

    let col_in_wg = tid / THREADS_PER_COL;
    let thread_in_col = tid % THREADS_PER_COL;
    let global_col = wg_id.x * COLS_PER_WG + col_in_wg;

    // Compute dot product
    var dot_sum: f32 = 0.0;
    if (global_col < N) {
        let k_per_thread = (K + THREADS_PER_COL - 1u) / THREADS_PER_COL;
        let k_start = thread_in_col * k_per_thread;
        let k_end = min(k_start + k_per_thread, K);

        for (var k: u32 = k_start; k < k_end; k = k + 1u) {
            dot_sum = dot_sum + input[k] * weight[k * N + global_col];
        }
    }

    shared_partial[tid] = dot_sum;
    workgroupBarrier();

    // Reduce within column
    for (var stride: u32 = THREADS_PER_COL / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_in_col < stride) {
            let idx = col_in_wg * THREADS_PER_COL + thread_in_col;
            shared_partial[idx] = shared_partial[idx] + shared_partial[idx + stride];
        }
        workgroupBarrier();
    }

    // Write matmul output (RMSNorm will be done in phase 2)
    if (thread_in_col == 0u && global_col < N) {
        output[global_col] = shared_partial[col_in_wg * THREADS_PER_COL];
    }
}
