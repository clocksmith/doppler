// qwen4_exp PLE gate (Qwen4ExpTextPLELayer.forward), one workgroup per (token, stream):
//   g = sum_d(key_normed[t, s, d] * query_normed[t, s, d]) / sqrt(STREAM_WIDTH)
//   g' = sqrt(max(|g|, 1e-6)) * sign(g)
//   gated[t, s, :] = sigmoid(g') * value[t, :]
// Products are rounded before summation; the divisor is f32-rounded like the
// source's tensor / math.sqrt(...). gate_input records g' for observation.

override WORKGROUP_SIZE: u32 = 256u;
override STREAM_COUNT: u32;
override STREAM_WIDTH: u32;

struct Uniforms {
    num_tokens: u32,
    group_stride: u32,
    divisor: f32,     // f32(sqrt(STREAM_WIDTH))
    clamp_min: f32,   // 1e-6
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> key_normed: array<f32>;    // [T, STREAM_COUNT * STREAM_WIDTH]
@group(0) @binding(2) var<storage, read> query_normed: array<f32>;  // [T, STREAM_COUNT * STREAM_WIDTH]
@group(0) @binding(3) var<storage, read> value: array<f32>;         // [T, STREAM_WIDTH]
@group(0) @binding(4) var<storage, read_write> gated: array<f32>;   // [T, STREAM_COUNT * STREAM_WIDTH]
@group(0) @binding(5) var<storage, read_write> gate_input: array<f32>; // [T, STREAM_COUNT]

var<workgroup> partial: array<f32, WORKGROUP_SIZE>;
var<workgroup> scale: f32;

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

fn divide_refined(x: f32, y: f32) -> f32 {
  let inverse = reciprocal_refined(y);
  let quotient = x * inverse;
  return fma(fma(-quotient, y, x), inverse, quotient);
}

fn sigmoid_refined(x: f32) -> f32 {
  let z = exp_refined(-abs(x));
  let reciprocal = reciprocal_refined(1.0 + z);
  return select(z * reciprocal, reciprocal, x >= 0.0);
}

fn sqrt_refined(x: f32) -> f32 {
  let estimate = inverseSqrt(x);
  let root = x * estimate;
  return fma(0.5 * estimate, fma(-root, root, x), root);
}

fn multiply_ordered(a: f32, b: f32) -> f32 {
  var product = 1.0;
  let factors = array<f32, 2>(a, b);
  for (var i = 0u; i < 2u; i = i + 1u) {
    product = fma(product, factors[i], 0.0);
  }
  return product;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let group = wid.y * u.group_stride + wid.x;
    if (group >= u.num_tokens * STREAM_COUNT) {
        return;
    }
    let token = group / STREAM_COUNT;
    let stream = group % STREAM_COUNT;
    let local = lid.x;
    let base = group * STREAM_WIDTH;

    var sum = 0.0;
    for (var d = local; d < STREAM_WIDTH; d = d + WORKGROUP_SIZE) {
        sum = sum + multiply_ordered(key_normed[base + d], query_normed[base + d]);
    }
    partial[local] = sum;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (local < stride) {
            partial[local] = partial[local] + partial[local + stride];
        }
        workgroupBarrier();
    }
    if (local == 0u) {
        let g = divide_refined(partial[0], u.divisor);
        let magnitude = sqrt_refined(max(abs(g), u.clamp_min));
        let signed = select(select(-magnitude, magnitude, g > 0.0), 0.0, g == 0.0);
        gate_input[token * STREAM_COUNT + stream] = signed;
        scale = sigmoid_refined(signed);
    }
    workgroupBarrier();
    let value_base = token * STREAM_WIDTH;
    for (var d = local; d < STREAM_WIDTH; d = d + WORKGROUP_SIZE) {
        gated[base + d] = multiply_ordered(scale, value[value_base + d]);
    }
}
