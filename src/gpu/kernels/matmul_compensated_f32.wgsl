// Opt-in single-F32 matrix multiplication with an internal compensated pair.
// Finite inputs/intermediates use exact product/addition residual recovery;
// the compensated alpha product is rounded once to the single-F32 output.
override TILE_SIZE: u32 = 16u;
override COMPENSATED_F32: bool = true;

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    transpose_b: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
var<workgroup> tile_a: array<f32, TILE_SIZE * TILE_SIZE>;
var<workgroup> tile_b: array<f32, TILE_SIZE * TILE_SIZE>;

// BEGIN GENERATED COMPENSATED STATE
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
// END GENERATED COMPENSATED STATE

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    let local_row = local_id.x;
    let local_col = local_id.y;
    let tile_idx = local_row * TILE_SIZE + local_col;
    var sum = vec2<f32>(0.0);
    let num_tiles = (u.K + TILE_SIZE - 1u) / TILE_SIZE;
    for (var t = 0u; t < num_tiles; t += 1u) {
        let a_col = t * TILE_SIZE + local_col;
        let b_row = t * TILE_SIZE + local_row;
        tile_a[tile_idx] = 0.0;
        tile_b[tile_idx] = 0.0;
        if (row < u.M && a_col < u.K) {
            tile_a[tile_idx] = A[row * u.K + a_col];
        }
        if (b_row < u.K && col < u.N) {
            if (u.transpose_b != 0u) {
                tile_b[tile_idx] = B[col * u.K + b_row];
            } else {
                tile_b[tile_idx] = B[b_row * u.N + col];
            }
        }
        workgroupBarrier();
        for (var k = 0u; k < TILE_SIZE; k += 1u) {
            let a = tile_a[local_row * TILE_SIZE + k];
            let b = tile_b[k * TILE_SIZE + local_col];
            let product = a * b;
            if (COMPENSATED_F32) {
                sum = pair_add(sum, vec2<f32>(product, product_residual(a, b, product)));
            } else {
                sum.x += product;
            }
        }
        workgroupBarrier();
    }
    if (row < u.M && col < u.N) {
        if (COMPENSATED_F32) {
            C[row * u.N + col] = pair_mul_scalar(sum, u.alpha).x;
        } else {
            C[row * u.N + col] = sum.x * u.alpha;
        }
    }
}
