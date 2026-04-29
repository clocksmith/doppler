// AUTO-GENERATED from src/gpu/kernels/rmsnorm.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
//
// F16 variant for reduced memory bandwidth when using F16 activations.
// The all-f16 lane keeps normalization arithmetic in f16.
// Weight buffer may be F16 or F32.
//
// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

enable f16;

override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;
override RMS_NORM_OFFSET: bool = false;   // Use (1 + weight) for Gemma models
override PRE_RESIDUAL: bool = false;      // Add residual BEFORE normalization: rmsnorm(input + residual)
override OUTPUT_PRENORM: bool = false;    // Write pre-norm sum to residual_sum_output binding
override WEIGHT_IS_F16: bool = false;     // Weight buffer packed as f16 pairs

struct Uniforms {
    size: u32,          // Hidden dimension
    num_tokens: u32,    // Number of tokens to process
    eps: f32,           // Epsilon for numerical stability
    has_residual: u32,  // 1 if residual input provided, 0 otherwise
    token_stride: u32,  // Workgroup rows per dispatch row
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> weight: array<u32>;   // F32 or packed F16
@group(0) @binding(3) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<storage, read> residual: array<f16>; // Optional residual
@group(0) @binding(5) var<storage, read_write> residual_sum_output: array<f16>; // Pre-norm sum output (PRE_RESIDUAL + OUTPUT_PRENORM)

var<workgroup> shared_sum: array<f16, MAX_WORKGROUP_SIZE>;

fn apply_weight(w: f16) -> f16 {
    if (RMS_NORM_OFFSET) {
        return f16(1.0) + w;
    }
    return w;
}

fn load_weight(idx: u32) -> f16 {
    if (WEIGHT_IS_F16) {
        let packed = weight[idx >> 1u];
        let pair = unpack2x16float(packed);
        return f16(select(pair.x, pair.y, (idx & 1u) == 1u));
    }
    return f16(bitcast<f32>(weight[idx]));
}

// Load input value, fusing residual add when PRE_RESIDUAL is active
fn load_input(base_offset: u32, idx: u32) -> f16 {
    let x = input[base_offset + idx];
    if (PRE_RESIDUAL) {
        return x + residual[base_offset + idx];
    }
    return x;
}

// Write pre-norm sum when OUTPUT_PRENORM is active (for downstream residual reuse)
fn write_prenorm(base_offset: u32, idx: u32, val: f16) {
    if (OUTPUT_PRENORM && PRE_RESIDUAL) {
        residual_sum_output[base_offset + idx] = val;
    }
}

fn token_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.y * max(u.token_stride, 1u) + wg_id.x;
}

// Main RMSNorm kernel - one workgroup per token
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let token_idx = token_index(wg_id);
    let thread_idx = local_id.x;
    let size = u.size;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * size;

    let elements_per_thread = (size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    var local_max: f16 = f16(0.0);
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            local_max = max(local_max, abs(load_input(base_offset, idx)));
        }
    }

    shared_sum[thread_idx] = local_max;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = max(shared_sum[thread_idx], shared_sum[thread_idx + stride]);
        }
        workgroupBarrier();
    }

    let row_scale = select(f16(1.0), shared_sum[0], shared_sum[0] > f16(0.0));
    workgroupBarrier();

    var local_sum_sq: f16 = f16(0.0);
    let inv_size = f16(1.0) / f16(size);
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = load_input(base_offset, idx) / row_scale;
            local_sum_sq = local_sum_sq + (x * x) * inv_size;
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

    let mean_sq = shared_sum[0];
    let eps_scaled = f16(u.eps) / (row_scale * row_scale);
    let inv_rms = (f16(1.0) / row_scale) / sqrt(mean_sq + eps_scaled);

    workgroupBarrier();

    // Apply normalization and weight
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = load_input(base_offset, idx);

            var result = x * inv_rms * apply_weight(load_weight(idx));

            // Add residual AFTER normalization
            if (u.has_residual == 1u) {
                result = result + residual[base_offset + idx];
            }

            // Write pre-norm sum for downstream residual reuse
            write_prenorm(base_offset, idx, x);

            output[base_offset + idx] = result;
        }
    }
}

// Optimized version for hidden size <= WORKGROUP_SIZE (single pass)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn rmsnorm_small_f16(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let token_idx = token_index(wg_id);
    let thread_idx = local_id.x;
    let size = u.size;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * size;

    // Each thread handles one element (for size <= 256)
    var x: f16 = f16(0.0);
    if (thread_idx < size) {
        x = load_input(base_offset, thread_idx);
    }

    shared_sum[thread_idx] = abs(x);
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            shared_sum[thread_idx] = max(shared_sum[thread_idx], shared_sum[thread_idx + stride]);
        }
        workgroupBarrier();
    }

    let row_scale = select(f16(1.0), shared_sum[0], shared_sum[0] > f16(0.0));
    let x_scaled = x / row_scale;
    let inv_size = f16(1.0) / f16(size);
    shared_sum[thread_idx] = (x_scaled * x_scaled) * inv_size;
    workgroupBarrier();

    // Parallel reduction
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride && thread_idx + stride < size) {
            shared_sum[thread_idx] = shared_sum[thread_idx] + shared_sum[thread_idx + stride];
        }
        workgroupBarrier();
    }

    // Compute inverse RMS
    let mean_sq = shared_sum[0];
    let eps_scaled = f16(u.eps) / (row_scale * row_scale);
    let inv_rms = (f16(1.0) / row_scale) / sqrt(mean_sq + eps_scaled);

    // Apply normalization
    if (thread_idx < size) {
        var result = x * inv_rms * apply_weight(load_weight(thread_idx));
        if (u.has_residual == 1u) {
            result = result + residual[base_offset + thread_idx];
        }
        write_prenorm(base_offset, thread_idx, x);
        output[base_offset + thread_idx] = result;
    }
}
