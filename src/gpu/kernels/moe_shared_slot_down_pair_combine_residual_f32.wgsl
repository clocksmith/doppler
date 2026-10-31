// AUTO-GENERATED from src/gpu/kernels/moe_shared_slot_down_pair_combine_f32.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// Geometry follows Pooled engine/wgsl/moe.js MOE_DEFAULT (6b4a24f7): staged input,
// row groups per workgroup, 2D (row group, route) dispatch and a separate fixed-order combine.
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
// Native GGUF shared-slot down projection fused with the slot combine and the layer's
// outer residual: moe_shared_slot_down_pair_combine_f32.wgsl with one more binding.
// One workgroup per (row group, token) runs all TOP_K routed slots and the shared slot
// (slot TOP_K) for its rows. Each slot's row dot uses the pair geometry of
// moe_shared_slot_down_pair_f32.wgsl (THREADS_PER_ROW invocations, fixed butterfly),
// then invocation 0 of each row applies moe_shared_slot_combine_f32.wgsl's expression
// in its order: sum_k w_k * y_k over k = 0 .. TOP_K - 1, plus the clamped
// sigmoid-gated shared slot. The rounded combine is then added to the residual row,
// as residual.wgsl adds the FFN output to it (unit scale). Per-slot outputs and the
// combine never leave the workgroup.
// Format branches are resolved by the ROUTED_FORMAT / SHARED_FORMAT pipeline constants.
enable subgroups;
requires subgroup_id;
requires packed_4x8_integer_dot_product;

override WORKGROUP_SIZE: u32 = 128u;
override THREADS_PER_ROW: u32 = 8u;
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
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> down_weights: array<u32>;
@group(0) @binding(3) var<storage, read> shared_down: array<u32>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> route_weights: array<f32>;
@group(0) @binding(6) var<storage, read> scalar_gate: array<f32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
@group(0) @binding(8) var<storage, read> residual: array<f32>;


fn load_words(is_shared: bool, base: u32, count: u32) -> array<u32, 17> {
    var w: array<u32, 17>;
    for (var i = 0u; i < 17u; i++) {
        if (i < count) {
            if (is_shared) { w[i] = shared_down[base + i]; } else { w[i] = down_weights[base + i]; }
        }
    }
    return w;
}
// Two consecutive native GGUF blocks start on a 4-byte boundary whenever the
// row holds an even number of blocks: Q4_0 18 B, Q4_1 20 B, Q5_0 22 B and
// Q8_0 34 B pairs are 9, 10, 11 and 17 aligned words. Each invocation owns one
// such pair (64 values) and reads it with whole-word loads; the stored bytes are
// unchanged. Format codes follow ggufBlockFormat: 0 Q4_0, 1 Q4_1, 2 Q8_0, 4 Q5_0.
fn pair_words(format: u32) -> u32 {
    switch format {
        case 0u: { return 9u; }
        case 1u: { return 10u; }
        case 2u: { return 17u; }
        default: { return 11u; }
    }
}
// Bytes 2..5 of a 4-byte-aligned window that starts two bytes into word i.
fn mid_word(w: array<u32, 17>, i: u32) -> u32 {
    return (w[i] >> 16u) | (w[i + 1u] << 16u);
}
fn nib_lo(word: u32) -> vec4<f32> { return vec4<f32>(unpack4xU8(word & 0x0F0F0F0Fu)); }
fn nib_hi(word: u32) -> vec4<f32> { return vec4<f32>(unpack4xU8((word >> 4u) & 0x0F0F0F0Fu)); }
fn q5_bits(qh: u32, first: u32) -> vec4<f32> {
    return vec4<f32>(((vec4<u32>(qh) >> (vec4<u32>(first) + vec4<u32>(0u, 1u, 2u, 3u))) & vec4<u32>(1u)) << vec4<u32>(4u));
}
// Dot of one block (32 values) with x[0..7] (value v -> x[v / 4][v % 4]).
fn block_dot(format: u32, scale: vec2<f32>, qh: u32, q0: u32, q1: u32, q2: u32, q3: u32,
             q4: u32, q5: u32, q6: u32, q7: u32, x: array<vec4<f32>, 16>, o: u32) -> f32 {
    if (format == 2u) {
        let s = dot(vec4<f32>(unpack4xI8(q0)), x[o]) + dot(vec4<f32>(unpack4xI8(q1)), x[o + 1u])
            + dot(vec4<f32>(unpack4xI8(q2)), x[o + 2u]) + dot(vec4<f32>(unpack4xI8(q3)), x[o + 3u])
            + dot(vec4<f32>(unpack4xI8(q4)), x[o + 4u]) + dot(vec4<f32>(unpack4xI8(q5)), x[o + 5u])
            + dot(vec4<f32>(unpack4xI8(q6)), x[o + 6u]) + dot(vec4<f32>(unpack4xI8(q7)), x[o + 7u]);
        return scale.x * s;
    }
    var bias = 8.0;
    var lo0 = nib_lo(q0); var lo1 = nib_lo(q1); var lo2 = nib_lo(q2); var lo3 = nib_lo(q3);
    var hi0 = nib_hi(q0); var hi1 = nib_hi(q1); var hi2 = nib_hi(q2); var hi3 = nib_hi(q3);
    if (format == 4u) {
        bias = 16.0;
        lo0 += q5_bits(qh, 0u); lo1 += q5_bits(qh, 4u); lo2 += q5_bits(qh, 8u); lo3 += q5_bits(qh, 12u);
        hi0 += q5_bits(qh, 16u); hi1 += q5_bits(qh, 20u); hi2 += q5_bits(qh, 24u); hi3 += q5_bits(qh, 28u);
    }
    if (format == 1u) {
        let s = dot(lo0, x[o]) + dot(lo1, x[o + 1u]) + dot(lo2, x[o + 2u]) + dot(lo3, x[o + 3u])
            + dot(hi0, x[o + 4u]) + dot(hi1, x[o + 5u]) + dot(hi2, x[o + 6u]) + dot(hi3, x[o + 7u]);
        let xs = (x[o] + x[o + 1u]) + (x[o + 2u] + x[o + 3u]) + (x[o + 4u] + x[o + 5u]) + (x[o + 6u] + x[o + 7u]);
        return scale.x * s + scale.y * dot(xs, vec4<f32>(1.0));
    }
    let b = vec4<f32>(bias);
    let s = dot(lo0 - b, x[o]) + dot(lo1 - b, x[o + 1u]) + dot(lo2 - b, x[o + 2u]) + dot(lo3 - b, x[o + 3u])
        + dot(hi0 - b, x[o + 4u]) + dot(hi1 - b, x[o + 5u]) + dot(hi2 - b, x[o + 6u]) + dot(hi3 - b, x[o + 7u]);
    return scale.x * s;
}
// Dot of one aligned block pair with x[0..15].
fn pair_dot(format: u32, w: array<u32, 17>, x: array<vec4<f32>, 16>) -> f32 {
    switch format {
        case 0u: {
            let a = block_dot(0u, vec2<f32>(unpack2x16float(w[0]).x, 0.0), 0u,
                mid_word(w, 0u), mid_word(w, 1u), mid_word(w, 2u), mid_word(w, 3u), 0u, 0u, 0u, 0u, x, 0u);
            let b = block_dot(0u, vec2<f32>(unpack2x16float(w[4]).y, 0.0), 0u,
                w[5], w[6], w[7], w[8], 0u, 0u, 0u, 0u, x, 8u);
            return a + b;
        }
        case 1u: {
            let a = block_dot(1u, unpack2x16float(w[0]), 0u, w[1], w[2], w[3], w[4], 0u, 0u, 0u, 0u, x, 0u);
            let b = block_dot(1u, unpack2x16float(w[5]), 0u, w[6], w[7], w[8], w[9], 0u, 0u, 0u, 0u, x, 8u);
            return a + b;
        }
        case 2u: {
            let a = block_dot(2u, vec2<f32>(unpack2x16float(w[0]).x, 0.0), 0u,
                mid_word(w, 0u), mid_word(w, 1u), mid_word(w, 2u), mid_word(w, 3u),
                mid_word(w, 4u), mid_word(w, 5u), mid_word(w, 6u), mid_word(w, 7u), x, 0u);
            let b = block_dot(2u, vec2<f32>(unpack2x16float(w[8]).y, 0.0), 0u,
                w[9], w[10], w[11], w[12], w[13], w[14], w[15], w[16], x, 8u);
            return a + b;
        }
        default: {
            let a = block_dot(4u, vec2<f32>(unpack2x16float(w[0]).x, 0.0), mid_word(w, 0u),
                mid_word(w, 1u), mid_word(w, 2u), mid_word(w, 3u), mid_word(w, 4u), 0u, 0u, 0u, 0u, x, 0u);
            let b = block_dot(4u, vec2<f32>(unpack2x16float(w[5]).y, 0.0), w[6],
                w[7], w[8], w[9], w[10], 0u, 0u, 0u, 0u, x, 8u);
            return a + b;
        }
    }
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


fn slot_dot(token: u32, slot: u32, row: u32, lane: u32, live: bool, is_shared: bool) -> f32 {
    let format = select(ROUTED_FORMAT, SHARED_FORMAT, is_shared);
    let width = select(u.intermediate_size, u.shared_size, is_shared);
    let input_pairs = width / 64u;
    let count = pair_words(format);
    let route = token * (TOP_K + 1u) + slot;
    var expert = 0u;
    if (live && !is_shared) { expert = indices[token * TOP_K + slot]; }
    var sum = 0.0;
    if (live && expert < u.num_experts) {
        let row_base = (expert * u.hidden_size + row) * input_pairs * count;
        for (var p = lane; p < input_pairs; p += THREADS_PER_ROW) {
            var x: array<vec4<f32>, 16>;
            for (var j = 0u; j < 16u; j++) { x[j] = input[route * (u.slot_stride / 4u) + p * 16u + j]; }
            sum += pair_dot(format, load_words(is_shared, row_base + p * count, count), x);
        }
    }
    for (var offset = THREADS_PER_ROW / 2u; offset > 0u; offset /= 2u) {
        sum += subgroupShuffleXor(sum, offset);
    }
    return sum;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(subgroup_id) subgroup: u32, @builtin(subgroup_invocation_id) sg_lane: u32,
        @builtin(subgroup_size) sg_size: u32) {
    let token = group.y;
    let linear = subgroup * sg_size + sg_lane;
    let lane = linear % THREADS_PER_ROW;
    let row = group.x * (WORKGROUP_SIZE / THREADS_PER_ROW) + linear / THREADS_PER_ROW;
    let live = token < u.num_tokens && row < u.hidden_size;
    var routed = 0.0;
    for (var slot = 0u; slot < TOP_K; slot++) {
        let y = slot_dot(token, slot, row, lane, live, false);
        if (live) { routed += route_weights[token * TOP_K + slot] * y; }
    }
    let shared_y = slot_dot(token, TOP_K, row, lane, live, true);
    if (live && lane == 0u) {
        let index = token * u.hidden_size + row;
        let combined = routed + clamp_product(sigmoid(scalar_gate[token]) * shared_y);
        output[index] = combined + residual[index];
    }
}
