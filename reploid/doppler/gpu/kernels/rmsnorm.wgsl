// RMSNorm Kernel with Fused Residual Add
//
// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
//
// Optionally fuses residual addition (POST-norm, for Gemma 3 sandwich norm):
// output = residual + RMSNorm(x) * weight
//
// Uses workgroup reduction for efficient mean calculation.

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    size: u32,          // Hidden dimension
    num_tokens: u32,    // Number of tokens to process
    eps: f32,           // Epsilon for numerical stability (typically 1e-5 or 1e-6)
    has_residual: u32,  // 1 if residual input provided, 0 otherwise
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;   // [size]
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> residual: array<f32>; // Optional residual input

// Shared memory for reduction
var<workgroup> shared_sum: array<f32, 256>;

// Main RMSNorm kernel - one workgroup per token
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let token_idx = wg_id.x;
    let thread_idx = local_id.x;
    let size = u.size;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * size;

    // Each thread computes partial sum of squares (on input only, NOT residual)
    var local_sum_sq: f32 = 0.0;
    let elements_per_thread = (size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = input[base_offset + idx];
            local_sum_sq = local_sum_sq + x * x;
        }
    }

    // Store local sum for reduction
    shared_sum[thread_idx] = local_sum_sq;
    workgroupBarrier();

    // Parallel reduction to compute total sum of squares
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }

    // Compute RMS
    let mean_sq = shared_sum[0] / f32(size);
    let rms = sqrt(mean_sq + u.eps);
    let inv_rms = 1.0 / rms;

    workgroupBarrier();

    // Apply normalization and weight, then add residual (POST-norm)
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = input[base_offset + idx];

            // Normalize and scale first
            var result = x * inv_rms * weight[idx];

            // Add residual AFTER normalization (Gemma 3 sandwich norm pattern)
            if (u.has_residual == 1u) {
                result = result + residual[base_offset + idx];
            }

            output[base_offset + idx] = result;
        }
    }
}

// Optimized version for hidden size <= 256 (single pass)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn rmsnorm_small(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let token_idx = wg_id.x;
    let thread_idx = local_id.x;
    let size = u.size;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * size;

    // Each thread handles one element (for size <= 256)
    // Read input only for RMS calculation (not residual)
    var x: f32 = 0.0;
    if (thread_idx < size) {
        x = input[base_offset + thread_idx];
    }

    // Sum of squares (on input only, NOT residual)
    shared_sum[thread_idx] = x * x;
    workgroupBarrier();

    // Parallel reduction
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride && thread_idx + stride < size) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }

    // Compute inverse RMS
    let mean_sq = shared_sum[0] / f32(size);
    let inv_rms = 1.0 / sqrt(mean_sq + u.eps);

    // Apply normalization, then add residual AFTER (POST-norm)
    if (thread_idx < size) {
        var result = x * inv_rms * weight[thread_idx];
        if (u.has_residual == 1u) {
            result = result + residual[base_offset + thread_idx];
        }
        output[base_offset + thread_idx] = result;
    }
}

// Version that also outputs the normalized input before weight multiplication
// Useful for some architectures that need both
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn rmsnorm_with_prenorm(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let token_idx = wg_id.x;
    let thread_idx = local_id.x;
    let size = u.size;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * size;
    let elements_per_thread = (size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // First pass: compute sum of squares (on input only, NOT residual)
    var local_sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = input[base_offset + idx];
            local_sum_sq = local_sum_sq + x * x;
        }
    }

    shared_sum[thread_idx] = local_sum_sq;
    workgroupBarrier();

    // Reduction
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }

    let mean_sq = shared_sum[0] / f32(size);
    let inv_rms = 1.0 / sqrt(mean_sq + u.eps);

    workgroupBarrier();

    // Second pass: write output with POST-norm residual
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = input[base_offset + idx];
            let normalized = x * inv_rms;
            var result = normalized * weight[idx];
            // Add residual AFTER normalization (POST-norm)
            if (u.has_residual == 1u) {
                result = result + residual[base_offset + idx];
            }
            output[base_offset + idx] = result;
        }
    }
}

// OPTIMIZED: Caches input to avoid double loads, adds residual POST-norm
// Uses shared memory to store input values between passes
var<workgroup> shared_cache: array<f32, 4608>;  // Max for hiddenSize=1152 × 4 elements/thread

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn rmsnorm_inplace_residual(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let token_idx = wg_id.x;
    let thread_idx = local_id.x;
    let size = u.size;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * size;
    let elements_per_thread = (size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // First pass: cache input and compute sum of squares (on input only)
    var local_sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = input[base_offset + idx];
            shared_cache[idx] = x;  // Cache input for second pass
            local_sum_sq = local_sum_sq + x * x;
        }
    }

    shared_sum[thread_idx] = local_sum_sq;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }

    let mean_sq = shared_sum[0] / f32(size);
    let inv_rms = 1.0 / sqrt(mean_sq + u.eps);

    workgroupBarrier();

    // Second pass: normalize cached input, then add residual POST-norm
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = shared_cache[idx];
            var result = x * inv_rms * weight[idx];
            // Add residual AFTER normalization (POST-norm)
            if (u.has_residual == 1u) {
                result = result + residual[base_offset + idx];
            }
            output[base_offset + idx] = result;
        }
    }
}
