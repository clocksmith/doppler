enable f16;
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
// Tiled two-F32 attention over an ordinary F16 K/V cache: one workgroup per
// head and query token walks the contiguous visible key range in tiles of
// WORKGROUP_SIZE - 1 keys. DOT_LANES lanes score each key, a tree finds the
// tile maximum, and every key's exponential runs in its own lane while the
// last lane evaluates the running-maximum rescale at the same callsite. The
// denominator and the value accumulators are rescaled once per tile.
override WORKGROUP_SIZE: u32 = 256u;
override DOT_LANES: u32 = 4u;
struct Uniforms {
    seq_len: u32,
    kv_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    kv_start: u32,
    sliding_window: u32,
    scale_hi: f32,
    scale_lo: f32,
    causal: u32,
    _pad: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> queries: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> keys: array<f16>;
@group(0) @binding(3) var<storage, read> values: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<vec2<f32>>;
var<workgroup> query: array<vec2<f32>, WORKGROUP_SIZE>;
var<workgroup> partials: array<vec2<f32>, WORKGROUP_SIZE>;
var<workgroup> present: array<u32, WORKGROUP_SIZE>;
// Scaled scores, then the same slots hold the tile's exponential weights.
var<workgroup> scores: array<vec2<f32>, WORKGROUP_SIZE>;
var<workgroup> rescale: vec2<f32>;
fn less(a: vec2<f32>, b: vec2<f32>) -> bool {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let head = group.x;
    let token = group.y;
    let kv_head = head / (u.num_heads / u.num_kv_heads);
    let query_base = (token * u.num_heads + head) * u.head_dim;
    let query_position = u.start_pos + token;
    // Causal and sliding-window visibility leave one contiguous key range.
    var key_end = u.kv_len;
    if (u.causal != 0u) {
        key_end = select(0u, min(key_end, query_position - u.kv_start + 1u), query_position >= u.kv_start);
    }
    var key_start = 0u;
    if (u.sliding_window != 0u && query_position + 1u > u.sliding_window + u.kv_start) {
        key_start = query_position + 1u - u.sliding_window - u.kv_start;
    }
    if (lane < u.head_dim) { query[lane] = queries[query_base + lane]; }
    let tile_keys = WORKGROUP_SIZE - 1u;
    let keys_per_round = WORKGROUP_SIZE / DOT_LANES;
    let slice = lane % DOT_LANES;
    let rescale_lane = WORKGROUP_SIZE - 1u;
    let scale = vec2<f32>(u.scale_hi, u.scale_lo);
    var maximum = vec2<f32>(0.0);
    var denominator = vec2<f32>(0.0);
    var has_value = false;
    var result = vec2<f32>(0.0);
    workgroupBarrier();
    for (var tile_start = key_start; tile_start < key_end; tile_start += tile_keys) {
        let count = min(tile_keys, key_end - tile_start);
        for (var round_start = 0u; round_start < count; round_start += keys_per_round) {
            let slot = round_start + lane / DOT_LANES;
            var partial = vec2<f32>(0.0);
            if (slot < count) {
                let key_base = ((tile_start + slot) * u.num_kv_heads + kv_head) * u.head_dim;
                for (var d = slice; d < u.head_dim; d += DOT_LANES) {
                    partial = pair_add(partial, pair_mul_scalar(query[d], f32(keys[key_base + d])));
                }
            }
            partials[lane] = partial;
            workgroupBarrier();
            for (var stride = DOT_LANES / 2u; stride > 0u; stride >>= 1u) {
                if (slice < stride) { partials[lane] = pair_add(partials[lane], partials[lane + stride]); }
                workgroupBarrier();
            }
            if (slice == 0u && slot < count) { scores[slot] = pair_mul(partials[lane], scale); }
        }
        workgroupBarrier();
        let scored = lane < count;
        var candidate = vec2<f32>(0.0);
        if (scored) { candidate = scores[lane]; }
        partials[lane] = candidate;
        present[lane] = u32(scored);
        workgroupBarrier();
        for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
            if (lane < stride && present[lane + stride] != 0u
                && (present[lane] == 0u || less(partials[lane], partials[lane + stride]))) {
                partials[lane] = partials[lane + stride];
                present[lane] = 1u;
            }
            workgroupBarrier();
        }
        let tile_maximum = partials[0];
        var next_maximum = tile_maximum;
        if (has_value && !less(maximum, tile_maximum)) { next_maximum = maximum; }
        workgroupBarrier();
        // One exponential callsite: key weights, plus the rescale in the last lane.
        let rescaling = lane == rescale_lane && has_value;
        var exponential = vec2<f32>(0.0);
        if (scored || rescaling) {
            exponential = pair_exp(pair_add(select(candidate, maximum, rescaling), -next_maximum));
        }
        if (scored) { scores[lane] = exponential; }
        if (lane == rescale_lane) { rescale = exponential; }
        partials[lane] = select(vec2<f32>(0.0), exponential, scored);
        workgroupBarrier();
        for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
            if (lane < stride) { partials[lane] = pair_add(partials[lane], partials[lane + stride]); }
            workgroupBarrier();
        }
        let alpha = rescale;
        denominator = pair_add(pair_mul(denominator, alpha), partials[0]);
        if (lane < u.head_dim) {
            result = pair_mul(result, alpha);
            for (var k = 0u; k < count; k++) {
                let value = f32(values[((tile_start + k) * u.num_kv_heads + kv_head) * u.head_dim + lane]);
                result = pair_add(result, pair_mul_scalar(scores[k], value));
            }
        }
        maximum = next_maximum;
        has_value = true;
        workgroupBarrier();
    }
    if (lane < u.head_dim) {
        var normalized = vec2<f32>(0.0);
        if (has_value) { normalized = pair_div(result, denominator); }
        output[query_base + lane] = normalized;
    }
}
