// Opt-in HC sigmoid + branch reduction. Preserve the separate operators'
// refined sigmoid, rounded sigmoid storage, rounded products, stream order
// and refined division. Cross-invocation exchanges retain both F32 boundaries.

override WORKGROUP_SIZE: u32 = 256u;
override STREAM_COUNT: u32;
override STREAM_WIDTH: u32;

struct Uniforms {
    size: u32,             // numTokens * STREAM_WIDTH
    dispatch_stride: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> normed: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Sigmoid output and product each cross an invocation boundary before use.
var<workgroup> rounded_mix: array<f32, WORKGROUP_SIZE>;
var<workgroup> rounded_products: array<f32, WORKGROUP_SIZE>;

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

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let index = gid.y * u.dispatch_stride + gid.x;
    let group_base = index - lid.x;
    let product_lane = (lid.x + 1u) % WORKGROUP_SIZE;
    let product_index = group_base + product_lane;
    let token_base = (index / STREAM_WIDTH) * STREAM_COUNT * STREAM_WIDTH + index % STREAM_WIDTH;
    let product_base = (product_index / STREAM_WIDTH) * STREAM_COUNT * STREAM_WIDTH + product_index % STREAM_WIDTH;
    var sum = 0.0;
    for (var stream = 0u; stream < STREAM_COUNT; stream = stream + 1u) {
        var mix = 0.0;
        if (index < u.size) {
            mix = sigmoid_refined(logits[token_base + stream * STREAM_WIDTH]);
        }
        rounded_mix[lid.x] = mix;
        workgroupBarrier();
        var product = 0.0;
        if (product_index < u.size) {
            product = rounded_mix[product_lane] * normed[product_base + stream * STREAM_WIDTH];
        }
        rounded_products[lid.x] = product;
        workgroupBarrier();
        sum = sum + rounded_products[(lid.x + WORKGROUP_SIZE - 1u) % WORKGROUP_SIZE];
        workgroupBarrier();
    }
    if (index < u.size) {
        output[index] = divide_refined(sum, f32(STREAM_COUNT));
    }
}
