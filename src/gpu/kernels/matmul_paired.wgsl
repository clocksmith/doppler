// Explicit interleaved two-F32 activations, unchanged F32/F16/Q4K weights.
override WORKGROUP_SIZE: u32 = 64u;
override WEIGHT_F16: bool = false;
override WEIGHT_Q4K: bool = false;
override TRANSPOSE_B: bool = true;
override SCALAR_OUTPUT: bool = false;
struct Uniforms {
    M: u32, N: u32, K: u32, grid_width: u32,
    alpha_hi: f32, alpha_lo: f32, _pad0: u32, _pad1: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> weight: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> partials: array<vec2<f32>, 64>;
// BEGIN GENERATED PAIRED ARITHMETIC
// Compensated recurrent-state arithmetic; finite operands and intermediate sums/products.
// Overflow propagates nonfinite values; this is not an extended-exponent format.
// This is a generator input, embedded verbatim in the four gated-delta kernels.
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
    if (bitcast<u32>(sum) == bitcast<u32>(a)) { return b; }
    if (bitcast<u32>(sum) == bitcast<u32>(b)) { return a; }
    let exponent = max(max((bitcast<u32>(a) >> 23u) & 255u, (bitcast<u32>(b) >> 23u) & 255u),
                       max((bitcast<u32>(sum) >> 23u) & 255u, 1u));
    if (exponent == 255u) { return 0.0; }
    let residual = add_u64(add_u64(aligned_significand(a, exponent), aligned_significand(b, exponent)),
                            negate_u64(aligned_significand(sum, exponent)));
    let negative = (residual.y & 0x80000000u) != 0u;
    let magnitude = select(residual, negate_u64(residual), negative);
    return scaled_significand(magnitude, i32(exponent) - 182, negative);
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


// Normalized two-component arithmetic. Integer residual recovery prevents
// relaxed floating-point reassociation from deleting a correction term.
fn add_preserving_underflow(a: f32, b: f32) -> f32 {
    let rounded = a + b;
    if ((bitcast<u32>(rounded) & 0x7fffffffu) == 0u) {
        return addition_residual(a, b, rounded);
    }
    return rounded;
}
fn pair_normalize(high: f32, low: f32) -> vec2<f32> {
    let rounded = high + low;
    let residual = addition_residual(high, low, rounded);
    if ((bitcast<u32>(rounded) & 0x7fffffffu) == 0u) {
        return vec2<f32>(residual, 0.0);
    }
    return vec2<f32>(rounded, residual);
}
fn pair_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let high = a.x + b.x;
    let error = addition_residual(a.x, b.x, high);
    let low = add_preserving_underflow(a.y, b.y);
    let joined = add_preserving_underflow(error, low);
    let tail = add_preserving_underflow(addition_residual(a.y, b.y, low), addition_residual(error, low, joined));
    let result = pair_normalize(high, joined);
    return pair_normalize(result.x, add_preserving_underflow(result.y, tail));
}
fn pair_mul_scalar(a: vec2<f32>, b: f32) -> vec2<f32> {
    let high = a.x * b;
    let low = a.y * b;
    return pair_add(vec2<f32>(high, product_residual(a.x, b, high)),
                    vec2<f32>(low, product_residual(a.y, b, low)));
}

// Paired elementary arithmetic. Requires compensated-f32.wgsl.inc.
// Finite normalized input pairs; finite intermediate sums/products. Division
// requires nonzero denominator, sqrt nonnegative input, and log positive input.
// Exp range reduction is bounded to finite binary32 exponent range. These are
// arithmetic domains, not model-selection policy. No extended exponent format.
fn pair_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return pair_add(pair_mul_scalar(a, b.x), pair_mul_scalar(a, b.y));
}
fn pair_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let first = a.x / b.x;
    var result = vec2<f32>(first, 0.0);
    for (var i = 0u; i < 2u; i++) {
        let residual = pair_add(a, -pair_mul(result, b));
        result = pair_add(result, vec2<f32>(residual.x / b.x, 0.0));
    }
    return result;
}
fn pair_sqrt(a: vec2<f32>) -> vec2<f32> {
    if (all(a == vec2<f32>(0.0))) { return a; }
    var result = vec2<f32>(sqrt(a.x), 0.0);
    for (var i = 0u; i < 2u; i++) {
        let residual = pair_add(a, -pair_mul(result, result));
        result = pair_add(result, pair_div(residual, pair_mul_scalar(result, 2.0)));
    }
    return result;
}
fn pair_exp(a: vec2<f32>) -> vec2<f32> {
    let ln2 = vec2<f32>(0.693147182464599609375, -1.904654323148236e-9);
    let exponent = i32(round(a.x * 1.4426950408889634));
    let reduced = pair_add(a, -pair_mul_scalar(ln2, f32(exponent)));
    var term = vec2<f32>(1.0, 0.0);
    var result = term;
    for (var i = 1u; i <= 20u; i++) {
        term = pair_div(pair_mul(term, reduced), vec2<f32>(f32(i), 0.0));
        result = pair_add(result, term);
    }
    return pair_normalize(ldexp(result.x, exponent), ldexp(result.y, exponent));
}
fn pair_log(a: vec2<f32>) -> vec2<f32> {
    let decomposition = frexp(a.x);
    var exponent = decomposition.exp;
    var mantissa = vec2<f32>(decomposition.fract, ldexp(a.y, -exponent));
    if (mantissa.x < 0.7071067811865476) {
        mantissa *= 2.0;
        exponent -= 1;
    }
    let one = vec2<f32>(1.0, 0.0);
    let z = pair_div(pair_add(mantissa, -one), pair_add(mantissa, one));
    let square = pair_mul(z, z);
    var power = z;
    var series = z;
    for (var i = 3u; i <= 41u; i += 2u) {
        power = pair_mul(power, square);
        series = pair_add(series, pair_div(power, vec2<f32>(f32(i), 0.0)));
    }
    let ln2 = vec2<f32>(0.693147182464599609375, -1.904654323148236e-9);
    return pair_add(pair_mul_scalar(series, 2.0), pair_mul_scalar(ln2, f32(exponent)));
}
fn pair_sigmoid(a: vec2<f32>) -> vec2<f32> {
    let one = vec2<f32>(1.0, 0.0);
    let nonnegative = a.x >= 0.0;
    // Keep one exponential and division callsite with the exact branch operands.
    let exponential = pair_exp(select(a, -a, nonnegative));
    return pair_div(select(exponential, one, nonnegative), pair_add(one, exponential));
}
fn pair_silu(a: vec2<f32>) -> vec2<f32> {
    return pair_mul(a, pair_sigmoid(a));
}
fn pair_softplus(a: vec2<f32>) -> vec2<f32> {
    let one = vec2<f32>(1.0, 0.0);
    let positive = a.x > 0.0;
    // Preserve the branch input and final addition without duplicating exp/log.
    let logarithm = pair_log(pair_add(one, pair_exp(select(a, -a, positive))));
    if (positive) { return pair_add(a, logarithm); }
    return logarithm;
}
fn pair_sin_cos(a: vec2<f32>) -> vec4<f32> {
    let half_pi = vec2<f32>(1.57079637050628662109375, -4.37113900018624283e-8);
    let quadrant = i32(round(a.x * 0.63661977236758134));
    // Subtract the large high product before combining the small terms. Making
    // a normalized pair for the whole quadrant*pi/2 first would discard bits
    // that become significant after this cancellation.
    let high_product = half_pi.x * f32(quadrant);
    let high_product_low = product_residual(half_pi.x, f32(quadrant), high_product);
    let first_reduction = pair_add(a, -vec2<f32>(high_product, high_product_low));
    let tail = vec2<f32>(half_pi.y, -1.7151244994428829e-15);
    let reduced = pair_add(first_reduction, -pair_mul_scalar(tail, f32(quadrant)));
    let negative_square = -pair_mul(reduced, reduced);
    var sin_term = reduced;
    var cosine_term = vec2<f32>(1.0, 0.0);
    var sine = sin_term;
    var cosine = cosine_term;
    for (var i = 1u; i <= 12u; i++) {
        sin_term = pair_div(pair_mul(sin_term, negative_square), vec2<f32>(f32((2u*i)*(2u*i+1u)), 0.0));
        cosine_term = pair_div(pair_mul(cosine_term, negative_square), vec2<f32>(f32((2u*i-1u)*(2u*i)), 0.0));
        sine = pair_add(sine, sin_term);
        cosine = pair_add(cosine, cosine_term);
    }
    let sector = u32(quadrant) & 3u;
    if (sector == 1u) { return vec4<f32>(cosine, -sine); }
    if (sector == 2u) { return vec4<f32>(-sine, -cosine); }
    if (sector == 3u) { return vec4<f32>(-cosine, sine); }
    return vec4<f32>(sine, cosine);
}
// END GENERATED PAIRED ARITHMETIC
fn weight_byte(index: u32) -> u32 { return (weight[index / 4u] >> ((index % 4u) * 8u)) & 255u; }
fn q4k_weight(row: u32, k: u32) -> f32 {
    let block = (row * (u.K / 256u) + k / 256u) * 144u;
    let scales = unpack2x16float(weight[block / 4u]);
    let local_k = k % 256u;
    let sub = local_k / 32u;
    var scale = 0u;
    var minimum = 0u;
    if (sub < 4u) {
        scale = weight_byte(block + 4u + sub) & 63u;
        minimum = weight_byte(block + 8u + sub) & 63u;
    } else {
        scale = (weight_byte(block + 8u + sub) & 15u) | ((weight_byte(block + sub) >> 6u) << 4u);
        minimum = (weight_byte(block + 8u + sub) >> 4u) | ((weight_byte(block + 4u + sub) >> 6u) << 4u);
    }
    let qbyte = weight_byte(block + 16u + (local_k / 64u) * 32u + local_k % 32u);
    let quant = (qbyte >> (((local_k / 32u) % 2u) * 4u)) & 15u;
    // Each F16 scale times these small integers is exactly F32. The stored
    // deployed-weight contract materializes their final difference once in F32.
    let positive = scales.x * f32(scale) * f32(quant);
    let negative = scales.y * f32(minimum);
    return positive - negative;
}
fn weight_value(row: u32, k: u32) -> f32 {
    if (WEIGHT_Q4K) { return q4k_weight(row, k); }
    var index = k * u.N + row;
    if (TRANSPOSE_B) { index = row * u.K + k; }
    if (WEIGHT_F16) { return unpack2x16float(weight[index / 2u])[index % 2u]; }
    return bitcast<f32>(weight[index]);
}
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) lane: vec3<u32>) {
    let index = group.x + group.y * u.grid_width;
    if (index >= u.M * u.N) { return; }
    let row = index / u.N;
    let column = index % u.N;
    var sum = vec2<f32>(0.0);
    for (var k = lane.x; k < u.K; k += WORKGROUP_SIZE) {
        sum = pair_add(sum, pair_mul_scalar(input[row * u.K + k], weight_value(column, k)));
    }
    partials[lane.x] = sum;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (lane.x < stride) { partials[lane.x] = pair_add(partials[lane.x], partials[lane.x + stride]); }
        workgroupBarrier();
    }
    if (lane.x == 0u) {
        let result = pair_mul(partials[0], vec2<f32>(u.alpha_hi, u.alpha_lo));
        if (SCALAR_OUTPUT) { output[index] = result.x; }
        else { output[index * 2u] = result.x; output[index * 2u + 1u] = result.y; }
    }
}
