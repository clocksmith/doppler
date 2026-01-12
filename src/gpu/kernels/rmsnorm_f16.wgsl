// RMSNorm Kernel with F16 Input/Output
//
// F16 variant for reduced memory bandwidth when using F16 activations.
// Intermediate computations (sum of squares, RMS) remain in F32 for precision.
// Weight buffer stays F32 (small size, precision matters).
//
// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

enable f16;

override WORKGROUP_SIZE: u32 = 256u;
override RMS_NORM_OFFSET: bool = false;   // Use (1 + weight) for Gemma models

struct Uniforms {
    size: u32,          // Hidden dimension
    num_tokens: u32,    // Number of tokens to process
    eps: f32,           // Epsilon for numerical stability
    has_residual: u32,  // 1 if residual input provided, 0 otherwise
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;   // Weights stay F32
@group(0) @binding(3) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<storage, read> residual: array<f16>; // Optional residual

// Shared memory for reduction (F32 for precision)
var<workgroup> shared_sum: array<f32, 256>;

fn apply_weight(w: f32) -> f32 {
    if (RMS_NORM_OFFSET) {
        return 1.0 + w;
    }
    return w;
}

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

    // Each thread computes partial sum of squares (promote to F32 for precision)
    var local_sum_sq: f32 = 0.0;
    let elements_per_thread = (size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = f32(input[base_offset + idx]);
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

    // Compute RMS (F32 for precision)
    let mean_sq = shared_sum[0] / f32(size);
    let rms = sqrt(mean_sq + u.eps);
    let inv_rms = 1.0 / rms;

    workgroupBarrier();

    // Apply normalization and weight, then add residual (POST-norm)
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = f32(input[base_offset + idx]);

            // Normalize and scale (compute in F32)
            var result = x * inv_rms * apply_weight(weight[idx]);

            // Add residual AFTER normalization
            if (u.has_residual == 1u) {
                result = result + f32(residual[base_offset + idx]);
            }

            // Convert back to F16 for output
            output[base_offset + idx] = f16(result);
        }
    }
}

// Optimized version for hidden size <= 256 (single pass)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn rmsnorm_small_f16(
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
    var x: f32 = 0.0;
    if (thread_idx < size) {
        x = f32(input[base_offset + thread_idx]);
    }

    // Sum of squares (F32 for precision)
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

    // Apply normalization (compute in F32, output F16)
    if (thread_idx < size) {
        var result = x * inv_rms * apply_weight(weight[thread_idx]);
        if (u.has_residual == 1u) {
            result = result + f32(residual[base_offset + thread_idx]);
        }
        output[base_offset + thread_idx] = f16(result);
    }
}
