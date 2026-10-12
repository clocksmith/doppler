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
    let phase = sincos_refined(angle);
    cos_values[index] = phase.x * magnitude;
    sin_values[index] = phase.y * magnitude;
}
