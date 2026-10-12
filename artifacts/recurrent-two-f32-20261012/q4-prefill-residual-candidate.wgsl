// Fused Q4_K Matmul Kernel — WideTile register pattern (F32 A, Q4_K B, F32 C)
//
// F32-output twin of fused_matmul_q4_widetile_f16.wgsl. Same register-tiling
// pattern and unpack4xU8-vectorized dequant — only the output dtype and the
// final cast differ. Kept as a separate file because the WGSL binding type
// is part of the kernel ABI; callers that want f32 output select this
// variant rather than this kernel coercing behind their back.
//
// See fused_matmul_q4_widetile_f16.wgsl for the full design rationale.

enable f16;

const QK_K: u32 = 256u;
const SUBBLOCK_SIZE: u32 = 32u;
const NUM_SUBBLOCK_PAIRS: u32 = 4u;
const U32S_PER_PAIR: u32 = 8u;

override TILE_M: u32 = 4u;
override TILE_N: u32 = 256u;

const MAX_TILE_M: u32 = 8u;

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    num_blocks_per_row: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Q4KBlock {
    d_dmin: u32,
    scales: array<u32, 3>,
    qs: array<u32, 32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B_q4k: array<Q4KBlock>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

var<workgroup> tileA: array<f32, MAX_TILE_M * QK_K>;

fn unpack_f16_lo(packed: u32) -> f32 { return unpack2x16float(packed).x; }
fn unpack_f16_hi(packed: u32) -> f32 { return unpack2x16float(packed).y; }

fn get_scale_byte(scales: array<u32, 3>, byte_idx: u32) -> u32 {
    let word_idx = byte_idx / 4u;
    let byte_in_word = byte_idx % 4u;
    return (scales[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn get_scale_min_k4(scales: array<u32, 3>, j: u32) -> vec2<u32> {
    var sc: u32;
    var mn: u32;
    if (j < 4u) {
        sc = get_scale_byte(scales, j) & 63u;
        mn = get_scale_byte(scales, j + 4u) & 63u;
    } else {
        let q_j = get_scale_byte(scales, j + 4u);
        let q_lo = get_scale_byte(scales, j - 4u);
        let q_hi = get_scale_byte(scales, j);
        sc = (q_j & 0xFu) | ((q_lo >> 6u) << 4u);
        mn = (q_j >> 4u) | ((q_hi >> 6u) << 4u);
    }
    return vec2<u32>(sc, mn);
}

fn load_a_vec4(row_local: u32, base_elem: u32) -> vec4<f32> {
    let a_row_base = row_local * QK_K;
    return vec4<f32>(
        tileA[a_row_base + base_elem + 0u],
        tileA[a_row_base + base_elem + 1u],
        tileA[a_row_base + base_elem + 2u],
        tileA[a_row_base + base_elem + 3u]
    );
}

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

fn compensated_product(a: vec4<f32>, w: vec4<f32>, high: vec4<f32>, low: ptr<function, vec4<f32>>) -> vec4<f32> {
    var result: vec4<f32>;
    for (var lane = 0u; lane < 4u; lane++) {
        let product = a[lane] * w[lane];
        let next = pair_add(vec2<f32>(high[lane], (*low)[lane]),
                            vec2<f32>(product, product_residual(a[lane], w[lane], product)));
        result[lane] = next.x;
        (*low)[lane] = next.y;
    }
    return result;
}

@compute @workgroup_size(TILE_N, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    if (TILE_M > MAX_TILE_M) { return; }

    let col_local = lid.x;
    let row_base = wg_id.y * TILE_M;
    let col_base = wg_id.x * TILE_N;
    let col = col_base + col_local;

    // Reduce each quantization block separately to bound F32 dependency depth.
    // Four local FMA lanes plus one outer scalar per row preserve bounded registers.
    var totals: array<vec2<f32>, MAX_TILE_M>;
    for (var i: u32 = 0u; i < TILE_M; i = i + 1u) {
        totals[i] = vec2<f32>(0.0);
    }

    let num_blocks = u.num_blocks_per_row;

    for (var b: u32 = 0u; b < num_blocks; b = b + 1u) {
        var results: array<vec4<f32>, MAX_TILE_M>;
        var corrections: array<vec4<f32>, MAX_TILE_M>;
        for (var m = 0u; m < TILE_M; m++) { results[m] = vec4<f32>(0.0); corrections[m] = vec4<f32>(0.0); }
        let total_a_elems = TILE_M * QK_K;
        for (var idx: u32 = col_local; idx < total_a_elems; idx = idx + TILE_N) {
            let load_row_local = idx / QK_K;
            let load_k = idx % QK_K;
            let global_row = row_base + load_row_local;
            let global_k = b * QK_K + load_k;
            if (global_row < u.M && global_k < u.K) {
                tileA[idx] = A[global_row * u.K + global_k];
            } else {
                tileA[idx] = 0.0;
            }
        }
        workgroupBarrier();

        if (col < u.N) {
            let block = B_q4k[col * num_blocks + b];
            let d = unpack_f16_lo(block.d_dmin);
            let dmin = unpack_f16_hi(block.d_dmin);

            for (var p: u32 = 0u; p < NUM_SUBBLOCK_PAIRS; p = p + 1u) {
                let sm_even = get_scale_min_k4(block.scales, 2u * p);
                let sm_odd = get_scale_min_k4(block.scales, 2u * p + 1u);
                let scale_even = d * f32(sm_even.x);
                let min_even_scalar = dmin * f32(sm_even.y);
                let scale_odd = d * f32(sm_odd.x);
                let min_odd_scalar = dmin * f32(sm_odd.y);
                let min_even = vec4<f32>(min_even_scalar);
                let min_odd = vec4<f32>(min_odd_scalar);

                let pair_base_even = p * 64u;
                let pair_base_odd = p * 64u + SUBBLOCK_SIZE;
                let qs_base = p * U32S_PER_PAIR;

                for (var uidx: u32 = 0u; uidx < U32S_PER_PAIR; uidx = uidx + 1u) {
                    let packed = block.qs[qs_base + uidx];
                    let lower_u32 = unpack4xU8(packed & 0x0F0F0F0Fu);
                    let upper_u32 = unpack4xU8((packed >> 4u) & 0x0F0F0F0Fu);
                    let w_even = scale_even * vec4<f32>(lower_u32) - min_even;
                    let w_odd = scale_odd * vec4<f32>(upper_u32) - min_odd;

                    let elem_even = pair_base_even + uidx * 4u;
                    let elem_odd = pair_base_odd + uidx * 4u;

                    for (var m: u32 = 0u; m < TILE_M; m = m + 1u) {
                        let a_even = load_a_vec4(m, elem_even);
                        let a_odd = load_a_vec4(m, elem_odd);
                        results[m] = compensated_product(a_even, w_even, results[m], &corrections[m]);
                        results[m] = compensated_product(a_odd, w_odd, results[m], &corrections[m]);
                    }
                }
            }
            for (var m = 0u; m < TILE_M; m++) {
                for (var lane = 0u; lane < 4u; lane++) {
                    totals[m] = pair_add(totals[m], vec2<f32>(results[m][lane], corrections[m][lane]));
                }
            }
        }
        workgroupBarrier();
    }

    if (col < u.N) {
        for (var m: u32 = 0u; m < TILE_M; m = m + 1u) {
            let row = row_base + m;
            if (row < u.M) {
                let scaled = pair_mul_scalar(totals[m], u.alpha);
                C[row * u.N + col] = scaled.x + scaled.y;
            }
        }
    }
}
