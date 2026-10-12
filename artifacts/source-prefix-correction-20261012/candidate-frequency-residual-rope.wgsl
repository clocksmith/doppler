override WORKGROUP_SIZE: u32 = 256u;
override USE_INVERSE_FREQUENCIES: bool = false;

struct Uniforms {
    max_seq_len: u32,
    rotary_dim: u32,
    frequency_base_dim: u32,
    scaling_type: u32,
    theta: f32,
    rope_scale: f32,
    yarn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    original_max_position: f32,
    dispatch_stride: u32,
    mrope_section_t: u32,
    mrope_section_h: u32,
    mrope_section_w: u32,
    frequency_offset: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> rope_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> cos_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> sin_values: array<f32>;

fn exp_refined(x: f32) -> f32 {
  if (abs(x) > 80.0) {
    return exp(x);
  }
  let n = round(x * 1.4426950408889634);
  var r = x;
  if (abs(x) >= 0.000000059604644775390625) {
    let bits = bitcast<u32>(abs(x));
    let mantissa = (bits & 0x007fffffu) | 0x00800000u;
    let shift = i32((bits >> 23u) & 255u) - 118;
    var input_fixed = 0u;
    if (shift >= 0) {
      input_fixed = mantissa << u32(shift);
    } else {
      let right = u32(-shift);
      input_fixed = (mantissa + (1u << (right - 1u))) >> right;
    }
    let power_fixed = u32(abs(n)) * 2977044472u;
    let residual = select(power_fixed - input_fixed, input_fixed - power_fixed, x >= 0.0);
    r = f32(bitcast<i32>(residual)) * 0.00000000023283064365386962890625;
  }
  var p = 0.0001984126984126984;
  p = fma(p, r, 0.001388888888888889);
  p = fma(p, r, 0.008333333333333333);
  p = fma(p, r, 0.041666666666666664);
  p = fma(p, r, 0.16666666666666666);
  p = fma(p, r, 0.5);
  p = fma(p, r, 1.0);
  p = fma(p, r, 1.0);
  return p * bitcast<f32>(u32(i32(n) + 127) << 23u);
}

fn reciprocal_refined(x: f32) -> f32 {
  let estimate = 1.0 / x;
  return fma(estimate, fma(-x, estimate, 1.0), estimate);
}

fn log_refined(x: f32) -> f32 {
  if (x <= 0.0 || x > 3.4028234663852886e38) {
    return log(x);
  }
  let bits = bitcast<u32>(x);
  var exponent = i32((bits >> 23u) & 255u) - 127;
  var mantissa = bitcast<f32>((bits & 0x007fffffu) | 0x3f800000u);
  if (mantissa > 1.4142135623730951) {
    mantissa = mantissa * 0.5;
    exponent = exponent + 1;
  }
  let s = (mantissa - 1.0) * reciprocal_refined(mantissa + 1.0);
  let squared = s * s;
  var p = 0.09090909090909091;
  p = fma(p, squared, 0.1111111111111111);
  p = fma(p, squared, 0.14285714285714285);
  p = fma(p, squared, 0.2);
  p = fma(p, squared, 0.3333333333333333);
  p = fma(p, squared, 1.0);
  let reduced = 2.0 * s * p;
  // Reconstruct exponent * ln(2) + reduced in signed Q32. Integer carry
  // and ties-to-even rounding make the final F32 rounding independent of
  // compiler contraction across the split-ln(2) expression. The rounded
  // ln(2) constant contributes < 6e-9 absolute error for normal F32 inputs.
  // exponent == 0 stays in F32 to preserve values arbitrarily close to one.
  if (exponent == 0) { return reduced; }
  let n = u32(abs(exponent));
  let bottom = n * (2977044472u & 65535u);
  let top = n * (2977044472u >> 16u);
  var lo = bottom + (top << 16u);
  var hi = (top >> 16u) + u32(lo < bottom);
  if (exponent < 0) {
    lo = 0u - lo;
    hi = ~hi + u32(lo == 0u);
  }
  let remainder = i32(round(reduced * 4294967296.0));
  let old_lo = lo;
  lo = lo + bitcast<u32>(remainder);
  hi = hi + select(0xffffffffu, 0u, remainder >= 0) + u32(lo < old_lo);
  let negative = (hi & 0x80000000u) != 0u;
  if (negative) {
    lo = 0u - lo;
    hi = ~hi + u32(lo == 0u);
  }
  let leading = select(firstLeadingBit(lo), 32u + firstLeadingBit(hi), hi != 0u);
  let shift = leading - 23u;
  var mant = (lo >> shift) | (hi << (32u - shift));
  let discarded = lo & ((1u << shift) - 1u);
  let halfway = 1u << (shift - 1u);
  mant = mant + u32(discarded > halfway || (discarded == halfway && (mant & 1u) != 0u));
  let magnitude = f32(mant) * bitcast<f32>((shift + 95u) << 23u);
  return select(magnitude, -magnitude, negative);
}


// Signed Q31 retains a remainder in [-pi/4, pi/4] across compiler contraction.
// The rounded pi/2 constant adds less than 2e-7 error through 4096 positions.
fn sincos_refined(angle: f32) -> vec2<f32> {
  // Above this range the Q31 constant error can exceed F32 trig accuracy.
  // Preserve the existing builtin path for those larger arguments.
  if (abs(angle) > 16384.0) { return vec2<f32>(cos(angle), sin(angle)); }
  let quadrant = round(angle * 0.6366197723675814);
  var reduced = angle;
  if (abs(angle) >= 0.000000059604644775390625) {
    let bits = bitcast<u32>(abs(angle));
    let mantissa = (bits & 0x007fffffu) | 0x00800000u;
    let shift = i32((bits >> 23u) & 255u) - 119;
    var input_fixed = 0u;
    if (shift >= 0) { input_fixed = mantissa << u32(shift); }
    else {
      let right = u32(-shift);
      input_fixed = (mantissa + (1u << (right - 1u))) >> right;
    }
    let power_fixed = u32(abs(quadrant)) * 3373259426u;
    let residual = select(power_fixed - input_fixed, input_fixed - power_fixed, angle >= 0.0);
    reduced = f32(bitcast<i32>(residual)) * 0.0000000004656612873077392578125;
  }
  let square = fma(reduced, reduced, 0.0);
  var sp = 1.6059043836821613e-10;
  sp = fma(sp, square, -2.505210838544172e-8);
  sp = fma(sp, square, 2.7557319223985893e-6);
  sp = fma(sp, square, -0.0001984126984126984);
  sp = fma(sp, square, 0.008333333333333333);
  sp = fma(sp, square, -0.16666666666666666);
  let sine = fma(reduced * square, sp, reduced);
  var cp = 2.08767569878681e-9;
  cp = fma(cp, square, -2.755731922398589e-7);
  cp = fma(cp, square, 0.0000248015873015873);
  cp = fma(cp, square, -0.001388888888888889);
  cp = fma(cp, square, 0.041666666666666664);
  cp = fma(cp, square, -0.5);
  let cosine = fma(square, cp, 1.0);
  switch (u32(i32(quadrant) & 3)) {
    case 1u: { return vec2<f32>(-sine, cosine); }
    case 2u: { return vec2<f32>(-cosine, -sine); }
    case 3u: { return vec2<f32>(sine, -cosine); }
    default: { return vec2<f32>(cosine, sine); }
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



// A compensated F32 temporary is reduced to the original F32 table layout.
fn pair_normalize(high: f32, low: f32) -> vec2<f32> {
  let rounded = high + low;
  return vec2<f32>(rounded, addition_residual(high, low, rounded));
}
fn pair_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let high = a.x + b.x;
  return pair_normalize(high, addition_residual(a.x, b.x, high) + (a.y + b.y));
}
fn pair_multiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let high = a.x * b.x;
  let low = product_residual(a.x, b.x, high) + (a.x * b.y + a.y * b.x) + a.y * b.y;
  return pair_normalize(high, low);
}
fn pair_divide(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let high = a.x / b.x;
  let remainder = pair_add(a, -pair_multiply(b, vec2<f32>(high, 0.0)));
  return pair_normalize(high, (remainder.x + remainder.y) / b.x);
}
fn logarithm_pair(value: f32) -> vec2<f32> {
  let bits = bitcast<u32>(value);
  var exponent = i32((bits >> 23u) & 255u) - 127;
  var mantissa = bitcast<f32>((bits & 0x7fffffu) | 0x3f800000u);
  if (mantissa > 1.4142135623730951) { mantissa *= 0.5; exponent += 1; }
  let s = pair_divide(pair_add(vec2<f32>(mantissa, 0.0), vec2<f32>(-1.0, 0.0)),
                            pair_add(vec2<f32>(mantissa, 0.0), vec2<f32>(1.0, 0.0)));
  let squared = pair_multiply(s, s);
  var polynomial = vec2<f32>(0.03703703731298447, -2.759474315716659e-10);
  polynomial = pair_add(pair_multiply(polynomial, squared), vec2<f32>(0.03999999910593033, 8.940696516468449e-10));
  polynomial = pair_add(pair_multiply(polynomial, squared), vec2<f32>(0.043478261679410934, -8.098457460192776e-10));
  polynomial = pair_add(pair_multiply(polynomial, squared), vec2<f32>(0.0476190485060215, -8.869738832295582e-10));
  polynomial = pair_add(pair_multiply(polynomial, squared), vec2<f32>(0.05263157933950424, -3.9213582381236733e-10));
  polynomial = pair_add(pair_multiply(polynomial, squared), vec2<f32>(0.05882352963089943, -2.1913472425527658e-10));
  polynomial = pair_add(pair_multiply(polynomial, squared), vec2<f32>(0.06666667014360428, -3.47693762670076e-9));
  polynomial = pair_add(pair_multiply(polynomial, squared), vec2<f32>(0.07692307978868484, -2.8656079731348427e-9));
  polynomial = pair_add(pair_multiply(polynomial, squared), vec2<f32>(0.09090909361839294, -2.709302115988521e-9));
  polynomial = pair_add(pair_multiply(polynomial, squared), vec2<f32>(0.1111111119389534, -8.278422947149977e-10));
  polynomial = pair_add(pair_multiply(polynomial, squared), vec2<f32>(0.1428571492433548, -6.38621200366174e-9));
  polynomial = pair_add(pair_multiply(polynomial, squared), vec2<f32>(0.20000000298023224, -2.9802322831784522e-9));
  polynomial = pair_add(pair_multiply(polynomial, squared), vec2<f32>(0.3333333432674408, -9.934107758624577e-9));
  polynomial = pair_add(pair_multiply(polynomial, squared), vec2<f32>(1.0, 0.0));
  let reduced = pair_multiply(vec2<f32>(2.0, 0.0), pair_multiply(s, polynomial));
  return pair_add(pair_multiply(vec2<f32>(f32(exponent), 0.0), vec2<f32>(0.6931471824645996, -1.9046542121259336e-9)), reduced);
}
fn exponential_pair(value: vec2<f32>) -> vec2<f32> {
  let exponent = round(value.x * 1.4426950408889634);
  let reduced = pair_add(value, -pair_multiply(vec2<f32>(exponent, 0.0), vec2<f32>(0.6931471824645996, -1.9046542121259336e-9)));
  var polynomial = vec2<f32>(1.147074536050896e-11, 2.372207689231238e-19);
  polynomial = pair_add(pair_multiply(polynomial, reduced), vec2<f32>(1.6059044372074283e-10, -5.352526511562726e-18));
  polynomial = pair_add(pair_multiply(polynomial, reduced), vec2<f32>(2.0876755879584152e-9, 1.1082839809204342e-16));
  polynomial = pair_add(pair_multiply(polynomial, reduced), vec2<f32>(2.5052107943679403e-8, 4.4176230446483665e-16));
  polynomial = pair_add(pair_multiply(polynomial, reduced), vec2<f32>(2.755731998149713e-7, -7.575112209051195e-15));
  polynomial = pair_add(pair_multiply(polynomial, reduced), vec2<f32>(0.0000027557318844628753, 3.793571224297229e-14));
  polynomial = pair_add(pair_multiply(polynomial, reduced), vec2<f32>(0.000024801587642286904, -3.40699609366682e-13));
  polynomial = pair_add(pair_multiply(polynomial, reduced), vec2<f32>(0.00019841270113829523, -2.725596874933456e-12));
  polynomial = pair_add(pair_multiply(polynomial, reduced), vec2<f32>(0.0013888889225199819, -3.3631094437103215e-11));
  polynomial = pair_add(pair_multiply(polynomial, reduced), vec2<f32>(0.008333333767950535, -4.34617203337595e-10));
  polynomial = pair_add(pair_multiply(polynomial, reduced), vec2<f32>(0.0416666679084301, -1.2417634698280722e-9));
  polynomial = pair_add(pair_multiply(polynomial, reduced), vec2<f32>(0.1666666716337204, -4.967053879312289e-9));
  polynomial = pair_add(pair_multiply(polynomial, reduced), vec2<f32>(0.5, 0.0));
  polynomial = pair_add(pair_multiply(polynomial, reduced), vec2<f32>(1.0, 0.0));
  polynomial = pair_add(pair_multiply(polynomial, reduced), vec2<f32>(1.0, 0.0));
  return vec2<f32>(ldexp(polynomial.x, i32(exponent)), ldexp(polynomial.y, i32(exponent)));
}
fn compensated_phase(theta: f32, dimension: u32, base_dimension: u32, position: f32) -> vec2<f32> {
  let exponent = pair_divide(vec2<f32>(-f32(dimension * 2u), 0.0), vec2<f32>(f32(base_dimension), 0.0));
  let frequency = exponential_pair(pair_multiply(exponent, logarithm_pair(theta)));
  let angle = pair_multiply(vec2<f32>(position, 0.0), frequency);
  let phase = sincos_refined(angle.x);
  // In the tested range the neglected quadratic term is below F32 rounding.
  return vec2<f32>(phase.x - phase.y * angle.y, phase.y + phase.x * angle.y);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_dim = u.rotary_dim / 2u;
    let index = gid.y * u.dispatch_stride + gid.x;
    let count = u.max_seq_len * half_dim;
    if (index >= count) {
        return;
    }
    let position = index / half_dim;
    let dimension = index % half_dim;
    let exponent = f32(dimension * 2u) / f32(u.frequency_base_dim);
    var frequency: f32;
    if (USE_INVERSE_FREQUENCIES) {
        frequency = bitcast<f32>(rope_data[u.frequency_offset + dimension]);
    } else {
        frequency = exp_refined(-exponent * log_refined(u.theta));
    }
    var scale = u.rope_scale;
    var magnitude = 1.0;
    var rope_position = f32(position);
    if (u.scaling_type == 1u) {
        let wavelength = 2.0 * 3.141592653589793 / frequency;
        let low_threshold = u.original_max_position / u.yarn_beta_slow;
        let high_threshold = u.original_max_position / u.yarn_beta_fast;
        if (wavelength < high_threshold) {
            scale = 1.0;
        } else if (wavelength > low_threshold) {
            scale = u.yarn_factor;
        } else {
            let mix = (wavelength - high_threshold) / (low_threshold - high_threshold);
            scale = 1.0 + (u.yarn_factor - 1.0) * mix;
        }
    } else if (u.scaling_type == 2u) {
        scale = bitcast<f32>(rope_data[dimension]);
        magnitude = sqrt(
            1.0 + log(f32(u.max_seq_len) / u.original_max_position)
                / log(u.original_max_position)
        );
    } else if (u.scaling_type == 3u) {
        let temporal_end = u.mrope_section_t;
        let height_end = temporal_end + u.mrope_section_h;
        var axis = 2u;
        if (dimension < temporal_end) {
            axis = 0u;
        } else if (dimension < height_end) {
            axis = 1u;
        }
        rope_position = f32(rope_data[axis * u.max_seq_len + position]);
    }
    let angle = (rope_position / scale) * frequency;
    var phase = sincos_refined(angle);
    // Preserve declared scaled/inverse-frequency paths outside this experiment.
    if (!USE_INVERSE_FREQUENCIES && u.scaling_type == 0u && scale == 1.0
      && u.theta >= 1.0 && u.theta <= 1.0e30 && rope_position <= 16384.0) {
      phase = compensated_phase(u.theta, dimension, u.frequency_base_dim, rope_position);
    }
    cos_values[index] = phase.x * magnitude;
    sin_values[index] = phase.y * magnitude;
}
