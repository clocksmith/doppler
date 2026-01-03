// RMSNorm Kernel with Fused Residual Add
//
// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
//
// Optionally fuses residual addition (POST-norm, for Gemma 3 sandwich norm):
// output = residual + RMSNorm(x) * weight
//
// Uses workgroup reduction for efficient mean calculation.
// Subgroup variants use subgroupAdd() for 3-5x faster reductions.
//
// Uses override constants for compile-time feature selection:
// - RMS_NORM_OFFSET: Use (1 + weight) pattern (Gemma models)
// - HAS_RESIDUAL: Enable post-norm residual addition
// - OUTPUT_PRENORM: Output pre-normalized values (before weight multiplication)

enable subgroups;

// =============================================================================
// Override Constants (compile-time configuration)
// =============================================================================

override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;

// Feature flags - compiler eliminates dead branches
override RMS_NORM_OFFSET: bool = false;   // Use (1 + weight) for Gemma models
override HAS_RESIDUAL: bool = false;      // Add residual after normalization
override OUTPUT_PRENORM: bool = false;    // Output prenorm values (not yet used)
override WEIGHT_IS_F16: bool = false;     // Weight buffer packed as f16 pairs

const MAX_SUBGROUPS: u32 = 32u;  // Support up to 32 subgroups per workgroup

// =============================================================================
// Uniforms (per-dispatch)
// =============================================================================

struct Uniforms {
    size: u32,          // Hidden dimension
    num_tokens: u32,    // Number of tokens to process
    eps: f32,           // Epsilon for numerical stability (typically 1e-5 or 1e-6)
    has_residual: u32,  // Runtime flag: 1 = add residual after norm
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<u32>;   // [size] as f32 or packed f16
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> residual: array<f32>; // Optional residual input

// =============================================================================
// Shared Memory
// =============================================================================

var<workgroup> shared_sum: array<f32, MAX_WORKGROUP_SIZE>;

// =============================================================================
// Helper Functions
// =============================================================================

// Apply weight with optional offset (Gemma uses 1 + weight pattern)
fn apply_weight(w: f32) -> f32 {
    if (RMS_NORM_OFFSET) {
        return 1.0 + w;
    } else {
        return w;
    }
}

fn load_weight(idx: u32) -> f32 {
    if (WEIGHT_IS_F16) {
        let packed = weight[idx >> 1u];
        let pair = unpack2x16float(packed);
        return select(pair.x, pair.y, (idx & 1u) == 1u);
    }
    return bitcast<f32>(weight[idx]);
}

// Check if residual should be added (compile-time OR runtime flag)
fn should_add_residual() -> bool {
    // HAS_RESIDUAL is compile-time override (for optimized paths)
    // u.has_residual is runtime flag (for flexibility)
    return HAS_RESIDUAL || (u.has_residual != 0u);
}

// =============================================================================
// Main Entry Point
// =============================================================================

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

            // Normalize and scale (with optional weight offset for Gemma)
            var result = x * inv_rms * apply_weight(load_weight(idx));

            // Add residual AFTER normalization (Gemma 3 sandwich norm pattern)
            if (should_add_residual()) {
                result = result + residual[base_offset + idx];
            }

            output[base_offset + idx] = result;
        }
    }
}

// =============================================================================
// Small Hidden Size Entry Point
// =============================================================================

// Optimized version for hidden size <= WORKGROUP_SIZE (single pass)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_small(
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
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
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
        var result = x * inv_rms * apply_weight(load_weight(thread_idx));
        if (should_add_residual()) {
            result = result + residual[base_offset + thread_idx];
        }
        output[base_offset + thread_idx] = result;
    }
}

// =============================================================================
// Cached Input Entry Point (avoids double loads)
// =============================================================================

// OPTIMIZED: Caches input to avoid double loads
// Uses shared memory to store input values between passes
var<workgroup> shared_cache: array<f32, 4608>;  // Max for hiddenSize=1152 × 4 elements/thread

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_cached(
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
            var result = x * inv_rms * apply_weight(load_weight(idx));
            // Add residual AFTER normalization (POST-norm)
            if (should_add_residual()) {
                result = result + residual[base_offset + idx];
            }
            output[base_offset + idx] = result;
        }
    }
}

// =============================================================================
// Subgroup-Accelerated Entry Points
// =============================================================================
// Use subgroupAdd() for 3-5x faster reductions vs tree reduction with barriers.
// Requires WebGPU subgroups feature.

var<workgroup> sg_partial_sums: array<f32, MAX_SUBGROUPS>;

// Subgroup-accelerated RMSNorm - one workgroup per token
// Uses subgroupAdd() to reduce 8 barriers → 1 barrier
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_subgroup(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let token_idx = wg_id.x;
    let thread_idx = local_id.x;
    let size = u.size;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * size;
    let subgroup_id = thread_idx / sg_size;
    let num_subgroups = (WORKGROUP_SIZE + sg_size - 1u) / sg_size;

    // Each thread computes partial sum of squares
    var local_sum_sq: f32 = 0.0;
    let elements_per_thread = (size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = input[base_offset + idx];
            local_sum_sq = local_sum_sq + x * x;
        }
    }

    // Phase 1: Subgroup reduction (no barrier needed!)
    let sg_sum = subgroupAdd(local_sum_sq);

    // Phase 2: Store subgroup results to shared memory
    if (sg_lane == 0u && subgroup_id < num_subgroups) {
        sg_partial_sums[subgroup_id] = sg_sum;
    }
    workgroupBarrier();  // Single barrier instead of 8!

    // Phase 3: Final reduction - thread 0 sums all subgroup partials
    // Then broadcast to all threads via shared memory
    if (thread_idx == 0u) {
        var sum: f32 = 0.0;
        for (var s = 0u; s < num_subgroups; s++) {
            sum += sg_partial_sums[s];
        }
        sg_partial_sums[0] = sum;
    }
    workgroupBarrier();
    let total_sum = sg_partial_sums[0];

    // Compute RMS (all threads now have the same value)
    let mean_sq = total_sum / f32(size);
    let inv_rms = 1.0 / sqrt(mean_sq + u.eps);

    // Apply normalization and weight
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = input[base_offset + idx];
            var result = x * inv_rms * apply_weight(load_weight(idx));

            if (should_add_residual()) {
                result = result + residual[base_offset + idx];
            }

            output[base_offset + idx] = result;
        }
    }
}

// Subgroup-accelerated RMSNorm for small hidden sizes (size <= 256)
// Each thread handles one element, uses subgroup reduction
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_small_subgroup(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let token_idx = wg_id.x;
    let thread_idx = local_id.x;
    let size = u.size;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * size;
    let subgroup_id = thread_idx / sg_size;
    let num_subgroups = (WORKGROUP_SIZE + sg_size - 1u) / sg_size;

    // Each thread handles one element
    var x: f32 = 0.0;
    var x_sq: f32 = 0.0;
    if (thread_idx < size) {
        x = input[base_offset + thread_idx];
        x_sq = x * x;
    }

    // Phase 1: Subgroup reduction
    let sg_sum = subgroupAdd(x_sq);

    // Phase 2: Store to shared memory
    if (sg_lane == 0u && subgroup_id < num_subgroups) {
        sg_partial_sums[subgroup_id] = sg_sum;
    }
    workgroupBarrier();

    // Phase 3: Final reduction - thread 0 sums, then broadcast via shared memory
    if (thread_idx == 0u) {
        var sum: f32 = 0.0;
        for (var s = 0u; s < num_subgroups; s++) {
            sum += sg_partial_sums[s];
        }
        sg_partial_sums[0] = sum;
    }
    workgroupBarrier();
    let total_sum = sg_partial_sums[0];

    // Compute inverse RMS
    let mean_sq = total_sum / f32(size);
    let inv_rms = 1.0 / sqrt(mean_sq + u.eps);

    // Apply normalization
    if (thread_idx < size) {
        var result = x * inv_rms * apply_weight(load_weight(thread_idx));
        if (should_add_residual()) {
            result = result + residual[base_offset + thread_idx];
        }
        output[base_offset + thread_idx] = result;
    }
}