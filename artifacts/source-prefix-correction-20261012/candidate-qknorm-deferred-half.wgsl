override DEFER_K_HALF_ROUNDING: bool = false;
override WORKGROUP_SIZE: u32 = 256u;
const MAX_WORKGROUP_SIZE: u32 = 256u;

override RMS_NORM_OFFSET: bool = false;
override Q_WEIGHT_IS_F16: bool = false;
override K_WEIGHT_IS_F16: bool = false;

struct Uniforms {
    q_rows: u32,
    k_rows: u32,
    head_dim: u32,
    row_stride: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> q_input: array<f32>;
@group(0) @binding(2) var<storage, read> q_weight: array<u32>;
@group(0) @binding(3) var<storage, read_write> q_output: array<f32>;
@group(0) @binding(4) var<storage, read> k_input: array<f32>;
@group(0) @binding(5) var<storage, read> k_weight: array<u32>;
@group(0) @binding(6) var<storage, read_write> k_output: array<f32>;

var<workgroup> shared_sum: array<f32, MAX_WORKGROUP_SIZE>;
var<workgroup> shared_correction: array<f32, MAX_WORKGROUP_SIZE>;

fn apply_weight(w: f32) -> f32 {
    if (RMS_NORM_OFFSET) {
        return 1.0 + w;
    }
    return w;
}

fn row_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.y * max(u.row_stride, 1u) + wg_id.x;
}

fn load_value(row: u32, idx: u32) -> f32 {
    if (row < u.q_rows) {
        return q_input[row * u.head_dim + idx];
    }
    let k_row = row - u.q_rows;
    return k_input[k_row * u.head_dim + idx];
}

fn load_weight(row: u32, idx: u32) -> f32 {
    if (row < u.q_rows) {
        if (Q_WEIGHT_IS_F16) {
            let packed = q_weight[idx >> 1u];
            let pair = unpack2x16float(packed);
            return select(pair.x, pair.y, (idx & 1u) == 1u);
        }
        return bitcast<f32>(q_weight[idx]);
    }
    if (K_WEIGHT_IS_F16) {
        let packed = k_weight[idx >> 1u];
        let pair = unpack2x16float(packed);
        return select(pair.x, pair.y, (idx & 1u) == 1u);
    }
    return bitcast<f32>(k_weight[idx]);
}

// Preserve the side of an F16 midpoint until the later F16 store.
// Only valid when this value proceeds directly to F16 materialization.
fn defer_half_rounding(high: f32, low: f32) -> f32 {
    let bits = bitcast<u32>(high);
    let exponent = (bits >> 23u) & 255u;
    if (exponent < 102u || exponent > 142u || (bitcast<u32>(low) & 0x7fffffffu) == 0u) { return high; }
    let shift = select(126u - exponent, 13u, exponent >= 113u);
    let significand = (bits & 0x007fffffu) | 0x00800000u;
    let mask = (1u << shift) - 1u;
    if ((significand & mask) != (1u << (shift - 1u))) { return high; }
    let same_sign = ((bits ^ bitcast<u32>(low)) & 0x80000000u) == 0u;
    return bitcast<f32>(select(bits - 1u, bits + 1u, same_sign));
}

fn store_value(row: u32, idx: u32, value: f32) {
    if (row < u.q_rows) {
        q_output[row * u.head_dim + idx] = value;
    } else {
        let k_row = row - u.q_rows;
        k_output[k_row * u.head_dim + idx] = value;
    }
}

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
    if (shift >= 64) { return bitcast<f32>(pb ^ 0x80000000u); }
    if (shift < 0) { return 0.0; }
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


@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let row = row_index(wg_id);
    let total_rows = u.q_rows + u.k_rows;
    if (row >= total_rows) {
        return;
    }

    let thread_idx = local_id.x;
    let size = u.head_dim;
    let elements_per_thread = (size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // Keep rounded sums observable before recovering their low parts. Without
    // the shared-memory boundary, compiler reassociation can erase compensation.
    var local_sum_sq = 0.0;
    var correction = 0.0;
    for (var i = 0u; i < elements_per_thread; i++) {
        let idx = thread_idx * elements_per_thread + i;
        var x = 0.0;
        if (idx < size) { x = load_value(row, idx); }
        let square = x * x;
        shared_sum[thread_idx] = local_sum_sq + square;
        workgroupBarrier();
        let rounded = shared_sum[thread_idx];
        let error = addition_residual(local_sum_sq, square, rounded);
        correction += product_residual(x, x, square) + error;
        local_sum_sq = rounded;
    }

    shared_sum[thread_idx] = local_sum_sq;
    shared_correction[thread_idx] = correction;
    workgroupBarrier();

    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        let a = shared_sum[thread_idx];
        let b = shared_sum[min(thread_idx + stride, WORKGROUP_SIZE - 1u)];
        if (thread_idx < stride) { shared_sum[thread_idx] = a + b; }
        workgroupBarrier();
        if (thread_idx < stride) {
            let rounded = shared_sum[thread_idx];
            let error = addition_residual(a, b, rounded);
            shared_correction[thread_idx] += shared_correction[thread_idx + stride] + error;
        }
        workgroupBarrier();
    }

    let mean_sq = shared_sum[0] / f32(size);
    let epsilon_sum = mean_sq + u.eps;
    let mean_error = shared_correction[0] / f32(size)
        + fma(-mean_sq, f32(size), shared_sum[0]) / f32(size);
    let variance_error = mean_error + addition_residual(mean_sq, u.eps, epsilon_sum);
    let estimate = inverseSqrt(epsilon_sum);
    let estimate_squared = estimate * estimate;
    let square_error = fma(estimate, estimate, -estimate_squared);
    let reciprocal_residual = fma(-epsilon_sum, estimate_squared, 1.0)
        - epsilon_sum * square_error - variance_error * estimate_squared;
    let inverse_error = (0.5 * estimate) * reciprocal_residual;

    workgroupBarrier();

    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = thread_idx * elements_per_thread + i;
        if (idx < size) {
            let x = load_value(row, idx);
            let w = apply_weight(load_weight(row, idx));
            let normalized = x * estimate;
            let normalization_error = product_residual(x, estimate, normalized) + x * inverse_error;
            let weighted = normalized * w;
            let output_error = product_residual(normalized, w, weighted) + normalization_error * w;
            let rounded = weighted + output_error;
            let low = addition_residual(weighted, output_error, rounded);
            let result = rounded + low;
            let residual = addition_residual(rounded, low, result);
            if (DEFER_K_HALF_ROUNDING && row >= u.q_rows) {
                store_value(row, idx, defer_half_rounding(result, residual));
                continue;
            }
            store_value(row, idx, result);
        }
    }
}
