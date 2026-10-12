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
// - HAS_RESIDUAL: Compile-time override (reserved for codegen patches, not used by JS dispatch)
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
override PRE_RESIDUAL: bool = false;      // Add residual BEFORE normalization: rmsnorm(input + residual)
override OUTPUT_PRENORM: bool = false;    // Write pre-norm sum to residual_sum_output binding
override WEIGHT_IS_F16: bool = false;     // Weight buffer packed as f16 pairs

const MAX_SUBGROUPS: u32 = 32u;  // Support up to 32 subgroups per workgroup
const MAX_CACHE_SIZE: u32 = 4608u;  // Supports hiddenSize up to 1152 with 4 elements/thread

// =============================================================================
// Uniforms (per-dispatch)
// =============================================================================

struct Uniforms {
    size: u32,          // Hidden dimension
    num_tokens: u32,    // Number of tokens to process
    eps: f32,           // Epsilon for numerical stability (typically 1e-5 or 1e-6)
    has_residual: u32,  // Runtime flag: 1 = add residual after norm
    token_stride: u32,  // Workgroup rows per dispatch row
    output_scale: f32,  // Output epilogue scale
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<u32>;   // [size] as f32 or packed f16
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> residual: array<f32>; // Optional residual input
@group(0) @binding(5) var<storage, read_write> residual_sum_output: array<f32>; // Pre-norm sum output (PRE_RESIDUAL + OUTPUT_PRENORM)

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

// Check if residual should be added (runtime uniform only — JS dispatch controls this)
fn should_add_residual() -> bool {
    return u.has_residual != 0u;
}

// Load input value, fusing residual add when PRE_RESIDUAL is active
fn load_input(base_offset: u32, idx: u32) -> f32 {
    let x = input[base_offset + idx];
    if (PRE_RESIDUAL) {
        return x + residual[base_offset + idx];
    }
    return x;
}

// Write pre-norm sum when OUTPUT_PRENORM is active (for downstream residual reuse)
fn write_prenorm(base_offset: u32, idx: u32, val: f32) {
    if (OUTPUT_PRENORM && PRE_RESIDUAL) {
        residual_sum_output[base_offset + idx] = val;
    }
}

fn apply_output_scale(value: f32) -> f32 {
    return value * u.output_scale;
}

fn token_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.y * max(u.token_stride, 1u) + wg_id.x;
}

// =============================================================================
// Main Entry Point
// =============================================================================

// Main RMSNorm kernel - one workgroup per token
// Recover the rounding residual of an F32 sum from its IEEE representation.
// A wider floating-point type is not used. Two U32 words retain the exact
// aligned significands so floating-point reassociation cannot erase the error.
fn add_u64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low = a.x + b.x;
    return vec2<u32>(low, a.y + b.y + u32(low < a.x));
}
fn negate_u64(a: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(0u - a.x, ~a.y + u32(a.x == 0u));
}
fn aligned_significand(value: f32, exponent: u32) -> vec2<u32> {
    let bits = bitcast<u32>(value);
    let stored_exponent = (bits >> 23u) & 255u;
    let mantissa = (bits & 0x007fffffu) | select(0u, 0x00800000u, stored_exponent != 0u);
    let shift = exponent - max(stored_exponent, 1u);
    var magnitude = vec2<u32>(0u);
    if (shift == 0u) { magnitude.y = mantissa; }
    else if (shift < 32u) { magnitude = vec2<u32>(mantissa << (32u - shift), mantissa >> shift); }
    else if (shift < 64u) { magnitude.x = mantissa >> (shift - 32u); }
    return select(magnitude, negate_u64(magnitude), (bits & 0x80000000u) != 0u);
}
fn addition_residual(a: f32, b: f32, sum: f32) -> f32 {
    if (sum == a) { return b; }
    if (sum == b) { return a; }
    let exponent = max(max((bitcast<u32>(a) >> 23u) & 255u, (bitcast<u32>(b) >> 23u) & 255u),
                       max((bitcast<u32>(sum) >> 23u) & 255u, 1u));
    if (exponent == 255u) { return 0.0; }
    let residual = add_u64(add_u64(aligned_significand(a, exponent), aligned_significand(b, exponent)),
                            negate_u64(aligned_significand(sum, exponent)));
    let negative = (residual.y & 0x80000000u) != 0u;
    let magnitude = select(residual, negate_u64(residual), negative);
    let value = ldexp(f32(magnitude.x), i32(exponent) - 182);
    return select(value, -value, negative);
}


// Construct the binary32 result, including nearest-even subnormal rounding,
// without an arithmetic operation that may flush subnormal operands/results.
fn rounded_shift(value: vec2<u32>, shift: u32) -> u32 {
    if (shift == 0u) { return value.x; }
    if (shift >= 64u) { return 0u; }
    var quotient: u32;
    var above: bool;
    var tie: bool;
    if (shift < 32u) {
        quotient = (value.x >> shift) | (value.y << (32u - shift));
        let remainder = value.x & ((1u << shift) - 1u);
        let halfway = 1u << (shift - 1u);
        above = remainder > halfway; tie = remainder == halfway;
    } else if (shift == 32u) {
        quotient = value.y; above = value.x > 0x80000000u; tie = value.x == 0x80000000u;
    } else {
        let high_shift = shift - 32u;
        quotient = value.y >> high_shift;
        let remainder = value.y & ((1u << high_shift) - 1u);
        let halfway = 1u << (high_shift - 1u);
        above = remainder > halfway || (remainder == halfway && value.x != 0u);
        tie = remainder == halfway && value.x == 0u;
    }
    return quotient + u32(above || (tie && (quotient & 1u) != 0u));
}
fn scaled_significand(magnitude: vec2<u32>, exponent: i32, negative: bool) -> f32 {
    let sign = select(0u, 0x80000000u, negative);
    if (all(magnitude == vec2<u32>(0u))) { return bitcast<f32>(sign); }
    let top = select(firstLeadingBit(magnitude.x), firstLeadingBit(magnitude.y) + 32u, magnitude.y != 0u);
    var unbiased = exponent + i32(top);
    var bits: u32;
    if (unbiased >= -126) {
        var significand = select(magnitude.x << (23u - min(top, 23u)), rounded_shift(magnitude, top - min(top, 23u)), top > 23u);
        if (significand >= 0x01000000u) { significand >>= 1u; unbiased += 1; }
        bits = (u32(unbiased + 127) << 23u) | (significand & 0x007fffffu);
    } else {
        let shift = exponent + 149;
        if (shift >= 0) { bits = magnitude.x << u32(shift); }
        else { bits = rounded_shift(magnitude, u32(-shift)); }
    }
    return bitcast<f32>(sign | bits);
}

// Exact product residual from the 24-bit significands, avoiding a floating
// multiply/subtract identity that a relaxed backend may reassociate.
fn product_residual(a: f32, b: f32, product: f32) -> f32 {
    let ab = bitcast<u32>(a); let bb = bitcast<u32>(b); let pb = bitcast<u32>(product);
    let ae = (ab >> 23u) & 255u; let be = (bb >> 23u) & 255u; let pe = (pb >> 23u) & 255u;
    if (ae == 255u || be == 255u || pe == 255u || (ab & 0x7fffffffu) == 0u || (bb & 0x7fffffffu) == 0u) { return 0.0; }
    let am = (ab & 0x007fffffu) | select(0u, 0x00800000u, ae != 0u);
    let bm = (bb & 0x007fffffu) | select(0u, 0x00800000u, be != 0u);
    let pm = (pb & 0x007fffffu) | select(0u, 0x00800000u, pe != 0u);
    let al = am & 65535u; let ah = am >> 16u;
    let bl = bm & 65535u; let bh = bm >> 16u;
    let cross = al * bh + ah * bl;
    let exact = add_u64(vec2<u32>(al * bl, ah * bh), vec2<u32>(cross << 16u, cross >> 16u));
    if ((pb & 0x7fffffffu) == 0u) {
        return scaled_significand(exact, i32(max(ae, 1u)) + i32(max(be, 1u)) - 300, ((ab ^ bb) & 0x80000000u) != 0u);
    }
    let shift = i32(max(pe, 1u)) + 150 - i32(max(ae, 1u)) - i32(max(be, 1u));
    if (shift < 0 || shift >= 64) { return 0.0; }
    var rounded = vec2<u32>(0u);
    if (shift == 0) { rounded.x = pm; }
    else if (shift < 32) { rounded = vec2<u32>(pm << u32(shift), pm >> (32u - u32(shift))); }
    else { rounded.y = pm << (u32(shift) - 32u); }
    let residual = add_u64(exact, negate_u64(rounded));
    let negative = (residual.y & 0x80000000u) != 0u;
    let magnitude = select(residual, negate_u64(residual), negative);
    let sign = negative != (((ab ^ bb) & 0x80000000u) != 0u);
    return scaled_significand(magnitude, i32(max(ae, 1u)) + i32(max(be, 1u)) - 300, sign);
}

var<workgroup> shared_error: array<f32, MAX_WORKGROUP_SIZE>;
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

    // Each thread computes partial sum of squares
    // When PRE_RESIDUAL is active, sum is computed on (input + residual)
    var local_sum_sq: f32 = 0.0;
    var local_error: f32 = 0.0;
    let elements_per_thread = (size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = load_input(base_offset, idx);
            let product = x * x;
            let sum = local_sum_sq + product;
            local_error += fma(x, x, -product) + addition_residual(local_sum_sq, product, sum);
            local_sum_sq = sum;
        }
    }

    // Store local sum for reduction
    shared_sum[thread_idx] = local_sum_sq;
    shared_error[thread_idx] = local_error;
    workgroupBarrier();

    // Parallel reduction to compute total sum of squares
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (thread_idx < stride) {
            let left = shared_sum[thread_idx];
            let right = shared_sum[thread_idx + stride];
            let sum = left + right;
            shared_error[thread_idx] += shared_error[thread_idx + stride] + addition_residual(left, right, sum);
            shared_sum[thread_idx] = sum;
        }
        workgroupBarrier();
    }

    // Refine the inverse root without a separately rounded square root.
    // The FMA residual includes the rounding error in estimate * estimate.
    let mean_sq = shared_sum[0] / f32(size);
    let epsilon_sum = mean_sq + u.eps;
    let mean_error = shared_error[0] / f32(size)
        + fma(-mean_sq, f32(size), shared_sum[0]) / f32(size);
    let variance_error = mean_error + addition_residual(mean_sq, u.eps, epsilon_sum);
    let estimate = inverseSqrt(epsilon_sum);
    let estimate_squared = estimate * estimate;
    let square_error = fma(estimate, estimate, -estimate_squared);
    let reciprocal_residual = fma(-epsilon_sum, estimate_squared, 1.0)
        - epsilon_sum * square_error - variance_error * estimate_squared;
    let inverse_error = (0.5 * estimate) * reciprocal_residual;

    workgroupBarrier();

    // Apply normalization and weight
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = load_input(base_offset, idx);

            // Normalize and scale (with optional weight offset for Gemma)
            let w = apply_weight(load_weight(idx));
            let normalized = x * estimate;
            let normalization_error = product_residual(x, estimate, normalized) + x * inverse_error;
            let weighted = normalized * w;
            let output_error = product_residual(normalized, w, weighted) + normalization_error * w;
            let rounded = weighted + output_error;
            var result = rounded + addition_residual(weighted, output_error, rounded);

            // Add residual AFTER normalization (Gemma 3 sandwich norm pattern)
            if (should_add_residual()) {
                result = result + residual[base_offset + idx];
            }

            // Write pre-norm sum for downstream residual reuse
            write_prenorm(base_offset, idx, x);

            output[base_offset + idx] = apply_output_scale(result);
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
    let token_idx = token_index(wg_id);
    let thread_idx = local_id.x;
    let size = u.size;

    if (token_idx >= u.num_tokens) {
        return;
    }

    let base_offset = token_idx * size;

    // Each thread handles one element (for size <= 256)
    var x: f32 = 0.0;
    if (thread_idx < size) {
        x = load_input(base_offset, thread_idx);
    }

    // Sum of squares
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

    // Apply normalization
    if (thread_idx < size) {
        var result = x * inv_rms * apply_weight(load_weight(thread_idx));
        if (should_add_residual()) {
            result = result + residual[base_offset + thread_idx];
        }
        write_prenorm(base_offset, thread_idx, x);
        output[base_offset + thread_idx] = apply_output_scale(result);
    }
}

// =============================================================================
// Cached Input Entry Point (avoids double loads)
// =============================================================================

// OPTIMIZED: Caches input to avoid double loads
// Uses shared memory to store input values between passes
var<workgroup> shared_cache: array<f32, MAX_CACHE_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_cached(
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

    // First pass: cache input and compute sum of squares
    var local_sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = load_input(base_offset, idx);
            shared_cache[idx] = x;  // Cache for second pass (avoids reloading residual)
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

    // Second pass: normalize cached input
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = shared_cache[idx];
            var result = x * inv_rms * apply_weight(load_weight(idx));
            if (should_add_residual()) {
                result = result + residual[base_offset + idx];
            }
            write_prenorm(base_offset, idx, x);
            output[base_offset + idx] = apply_output_scale(result);
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
    let token_idx = token_index(wg_id);
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
            let x = load_input(base_offset, idx);
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
            let x = load_input(base_offset, idx);
            var result = x * inv_rms * apply_weight(load_weight(idx));

            if (should_add_residual()) {
                result = result + residual[base_offset + idx];
            }

            write_prenorm(base_offset, idx, x);
            output[base_offset + idx] = apply_output_scale(result);
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
    let token_idx = token_index(wg_id);
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
        x = load_input(base_offset, thread_idx);
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
        write_prenorm(base_offset, thread_idx, x);
        output[base_offset + thread_idx] = apply_output_scale(result);
    }
}
