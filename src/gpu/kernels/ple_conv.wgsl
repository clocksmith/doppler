// qwen4_exp PLE dilated depthwise short convolution and residual, one thread
// per channel over all tokens of the call (Qwen4ExpTextPLELayer._short_conv):
//   conv[t, c] = sum_k weight[c, k] * x[t - STATE_LEN + k * DILATION]
// where x is the normalized gated value, preceded by the owned state of the
// previous STATE_LEN values (zeros for a new sequence). Then
//   output[t, c] = streams[t, c] + (gated[t, c] + silu(conv[t, c]))
// in that rounding order. The state then holds the last STATE_LEN values.

override WORKGROUP_SIZE: u32 = 256u;
override KERNEL: u32;
override DILATION: u32;
override STATE_LEN: u32;

const MAX_STATE: u32 = 32u;

struct Uniforms {
    num_tokens: u32,
    channels: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> normed: array<f32>;       // [T, channels]
@group(0) @binding(2) var<storage, read> gated: array<f32>;        // [T, channels]
@group(0) @binding(3) var<storage, read> streams: array<f32>;      // [T, channels]
@group(0) @binding(4) var<storage, read> conv_weight: array<f32>;  // [channels, KERNEL]
@group(0) @binding(5) var<storage, read_write> state: array<f32>;  // [STATE_LEN, channels], oldest first
@group(0) @binding(6) var<storage, read_write> output: array<f32>; // [T, channels]

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

fn silu(x: f32) -> f32 {
  let z = exp_refined(-abs(x));
  let numerator = select(x * z, x, x >= 0.0);
  return numerator * reciprocal_refined(1.0 + z);
}

fn history(t: i32, channel: u32) -> f32 {
  if (t >= 0) {
    return normed[u32(t) * u.channels + channel];
  }
  return state[u32(i32(STATE_LEN) + t) * u.channels + channel];
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let channel = gid.x;
    if (channel >= u.channels || STATE_LEN > MAX_STATE || (KERNEL - 1u) * DILATION != STATE_LEN) {
        return;
    }
    let tokens = i32(u.num_tokens);
    for (var t = 0; t < tokens; t = t + 1) {
        var acc = 0.0;
        for (var k = 0u; k < KERNEL; k = k + 1u) {
            let source = t - i32(STATE_LEN) + i32(k * DILATION);
            acc = fma(conv_weight[channel * KERNEL + k], history(source, channel), acc);
        }
        let index = u32(t) * u.channels + channel;
        output[index] = streams[index] + (gated[index] + silu(acc));
    }
    var next: array<f32, MAX_STATE>;
    for (var j = 0u; j < STATE_LEN; j = j + 1u) {
        next[j] = history(tokens - i32(STATE_LEN) + i32(j), channel);
    }
    for (var j = 0u; j < STATE_LEN; j = j + 1u) {
        state[j * u.channels + channel] = next[j];
    }
}
