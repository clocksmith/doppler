// AUTO-GENERATED from src/gpu/kernels/codegen/q4_0_common.wgslinc.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// Shared-slot topology adapted from Pooled engine/wgsl/moe.js (6b4a24f7).
// Native GGUF packing, Doppler refined SiLU, and no outer residual.
// MIT License
//
// Copyright (c) 2026 Nehanth Narendrula
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
override WORKGROUP_SIZE: u32 = 64u;
override TOP_K: u32;
override ROUTED_FORMAT: u32;
override ROUTED_BYTES: u32;
override SHARED_FORMAT: u32;
override SHARED_BYTES: u32;
struct Uniforms {
    num_tokens: u32, hidden_size: u32, intermediate_size: u32, shared_size: u32,
    num_experts: u32, slot_stride: u32, clamp_max: f32, _pad: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
var<private> bank: u32;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gate_weights: array<u32>;
@group(0) @binding(3) var<storage, read> up_weights: array<u32>;
@group(0) @binding(4) var<storage, read> shared_gate: array<u32>;
@group(0) @binding(5) var<storage, read> shared_up: array<u32>;
@group(0) @binding(6) var<storage, read> indices: array<u32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
fn bank_word(index: u32) -> u32 {
    switch bank {
        case 0u: { return gate_weights[index]; }
        case 1u: { return up_weights[index]; }
        case 2u: { return shared_gate[index]; }
        default: { return shared_up[index]; }
    }
}
fn q40_byte(offset: u32) -> u32 {
    return (bank_word(offset / 4u) >> ((offset % 4u) * 8u)) & 255u;
}
fn packed_half(offset: u32) -> f32 {
    return unpack2x16float(q40_byte(offset) | (q40_byte(offset + 1u) << 8u)).x;
}
fn packed_signed_byte(offset: u32) -> i32 {
    return bitcast<i32>(q40_byte(offset) << 24u) >> 24;
}
fn q40_value(row: u32, k: u32, blocks: u32, format: u32, values: u32, bytes: u32) -> f32 {
    let offset = (row * blocks + k / values) * bytes;
    let local = k % values;
    if (format == 3u) {
        let section = (local % 128u) / 32u;
        let lane = local % 32u;
        let low = q40_byte(offset + (local / 128u) * 64u + (section % 2u) * 32u + lane);
        let high = q40_byte(offset + 128u + (local / 128u) * 32u + lane);
        let value = ((low >> ((section / 2u) * 4u)) & 15u) | (((high >> (section * 2u)) & 3u) << 4u);
        return packed_half(offset + 208u) * f32(packed_signed_byte(offset + 192u + local / 16u)) * f32(i32(value) - 32);
    }
    let scale = packed_half(offset);
    if (format == 2u) { return scale * f32(packed_signed_byte(offset + 2u + local)); }
    if (format == 4u) {
        // Q5_0: f16 scale, 32 high bits (one per value), 16 bytes of low nibbles.
        let high_bits = q40_byte(offset + 2u) | (q40_byte(offset + 3u) << 8u)
            | (q40_byte(offset + 4u) << 16u) | (q40_byte(offset + 5u) << 24u);
        let low = (q40_byte(offset + 6u + local % 16u) >> select(0u, 4u, local >= 16u)) & 15u;
        return scale * f32(i32(low | (((high_bits >> local) & 1u) << 4u)) - 16);
    }
    let data_offset = select(2u,4u,format == 1u);
    let packed = q40_byte(offset + data_offset + local % 16u);
    let shift = select(0u,4u,local >= 16u);
    let value = (packed >> shift) & 15u;
    if (format == 1u) { return scale * f32(value) + packed_half(offset + 2u); }
    return scale * f32(i32(value) - 8);
}

fn packed_word(offset: u32) -> u32 {
    let shift = (offset % 4u) * 8u;
    let low = bank_word(offset / 4u);
    if (shift == 0u) { return low; }
    return (low >> shift) | (bank_word(offset / 4u + 1u) << (32u - shift));
}
fn signed_bytes(word: u32) -> vec4<f32> {
    let w = bitcast<i32>(word);
    return vec4<f32>(f32((w << 24u) >> 24u), f32((w << 16u) >> 24u),
        f32((w << 8u) >> 24u), f32(w >> 24u));
}
fn quarter_term(row: u32, b: u32, quarter: u32, blocks: u32, format: u32, bytes: u32,
                xbase: u32) -> f32 {
    let first = select(quarter * 4u, quarter * 8u, format == 2u);
    let second = select(first + 16u, first + 4u, format == 2u);
    let x = xbase + b * 32u;
    let xa = vec4<f32>(input[x+first], input[x+first+1u], input[x+first+2u], input[x+first+3u]);
    let xb = vec4<f32>(input[x+second], input[x+second+1u], input[x+second+2u], input[x+second+3u]);
    let base = (row * blocks + b) * bytes;
    let scale = packed_half(base);
    if (format == 2u) {
        return scale * (dot(signed_bytes(packed_word(base+2u+first)), xa)
            + dot(signed_bytes(packed_word(base+2u+second)), xb));
    }
    if (format == 0u) {
        let word = packed_word(base + 2u + quarter * 4u);
        let shifts = vec4<u32>(0u, 8u, 16u, 24u);
        let lo = vec4<f32>((vec4<u32>(word) >> shifts) & vec4<u32>(15u)) - vec4<f32>(8.0);
        let hi = vec4<f32>((vec4<u32>(word >> 4u) >> shifts) & vec4<u32>(15u)) - vec4<f32>(8.0);
        return scale * (dot(lo, xa) + dot(hi, xb));
    }
    var sum = 0.0;
    for (var j = 0u; j < 4u; j++) {
        sum = fma(xa[j], q40_value(row, b*32u+first+j, blocks, format, 32u, bytes), sum);
        sum = fma(xb[j], q40_value(row, b*32u+second+j, blocks, format, 32u, bytes), sum);
    }
    return sum;
}
fn clamp_product(value: f32) -> f32 {
    if (u.clamp_max > 0.0) { return clamp(value, -u.clamp_max, u.clamp_max); }
    return value;
}
fn sigmoid(x: f32) -> f32 {
    let clamped = clamp(x, -15.0, 15.0);
    let z = exp_refined(-abs(clamped));
    let numerator = select(z, 1.0, clamped >= 0.0);
    let denominator = 1.0 + z;
    let inverse = reciprocal_refined(denominator);
    let quotient = numerator * inverse;
    return fma(fma(-quotient, denominator, numerator), inverse, quotient);
}

// Compensate the rounded numerator and quotient; retain the established gate clamp.
fn silu(x: f32) -> f32 {
    let z = exp_refined(-abs(clamp(x, -15.0, 15.0)));
 let numerator = select(x * z, x, x >= 0.0);
 let error = select(fma(x, z, -numerator), 0.0, x >= 0.0);
 let den = 1.0 + z;
 let inverse = reciprocal_refined(den);
 let q = numerator * inverse;
 return fma(fma(-q, den, numerator) + error, inverse, q);
}

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

var<workgroup> partial_gate: array<vec4<f32>, WORKGROUP_SIZE>;
var<workgroup> partial_up: array<vec4<f32>, WORKGROUP_SIZE>;
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let tiles = (max(u.intermediate_size, u.shared_size) + 3u) / 4u;
    let tile = group.x + group.y * grid.x;
    let route = tile / tiles;
    if (route >= u.num_tokens * (TOP_K + 1u)) { return; }
    let slot = route % (TOP_K + 1u);
    let token = route / (TOP_K + 1u);
    let is_shared = slot == TOP_K;
    let rows = select(u.intermediate_size, u.shared_size, is_shared);
    let row = (tile % tiles) * 4u;
    if (row >= rows) { return; }
    var expert = 0u;
    if (!is_shared) { expert = indices[token * TOP_K + slot]; }
    var gs = vec4<f32>(0.0); var us = vec4<f32>(0.0);
    if (expert < u.num_experts) {
        for (var b = lane / 4u; b < u.hidden_size / 32u; b += WORKGROUP_SIZE / 4u) {
            for (var r = 0u; r < 4u; r++) {
                if (row + r < rows) {
                    let er = expert * rows + row + r;
                    let format = select(ROUTED_FORMAT, SHARED_FORMAT, is_shared);
                    let bytes = select(ROUTED_BYTES, SHARED_BYTES, is_shared);
                    bank = select(0u, 2u, is_shared);
                    gs[r] += quarter_term(er, b, lane % 4u, u.hidden_size / 32u, format, bytes, token*u.hidden_size);
                    bank += 1u;
                    us[r] += quarter_term(er, b, lane % 4u, u.hidden_size / 32u, format, bytes, token*u.hidden_size);
                }
            }
        }
    }
    partial_gate[lane] = gs; partial_up[lane] = us;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (lane < stride) { partial_gate[lane] += partial_gate[lane+stride]; partial_up[lane] += partial_up[lane+stride]; }
        workgroupBarrier();
    }
    if (lane < 4u && row + lane < rows) {
        output[route*u.slot_stride+row+lane] = clamp_product(silu(partial_gate[0][lane]) * partial_up[0][lane]);
    }
}
