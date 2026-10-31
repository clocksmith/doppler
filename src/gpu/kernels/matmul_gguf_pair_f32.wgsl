// Native GGUF Q4_0 / Q8_0 GEMV over aligned block pairs. Two consecutive blocks of a
// row with an even block count start on a 4-byte boundary (Q4_0 9 words, Q8_0 17),
// and the odd block's scale and quants are word-aligned. Invocations [0, WG/2) take
// even blocks and [WG/2, WG) odd blocks, so every wave runs one decode path; four
// invocations share a block (quarters, as in the cooperative GEMV) and each reuses its
// activation slice across ROWS rows. Waves reduce with subgroupAdd and the wave sums
// are added in wave order. The stored bytes are the GGUF bytes.
// Quarter mapping adapted from Pooled engine/wgsl/coop.js (6b4a24f7).
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
enable subgroups;
requires subgroup_id;
requires packed_4x8_integer_dot_product;

override WORKGROUP_SIZE: u32 = 128u;
override ROWS: u32 = 1u;
override QUANT_FORMAT: u32 = 0u;   // ggufBlockFormat code: 0 Q4_0, 2 Q8_0
override BLOCK_VALUES: u32 = 32u;
override BLOCK_BYTES: u32 = 18u;

struct Uniforms {
    M: u32, N: u32, K: u32, alpha: f32,
    blocks: u32, a_offset: u32, b_row: u32, c_offset: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> wave_sums: array<vec4<f32>, WORKGROUP_SIZE>;

fn low_nibbles(w: u32) -> vec4<f32> { return vec4<f32>(unpack4xU8(w & 0x0F0F0F0Fu)) - vec4<f32>(8.0); }
fn high_nibbles(w: u32) -> vec4<f32> { return vec4<f32>(unpack4xU8((w >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0); }
fn signed_bytes(w: u32) -> vec4<f32> { return vec4<f32>(unpack4xI8(w)); }

// Quarter `q` of block `odd` of the pair starting at word `wp`, dotted with xa / xb.
fn quarter_dot(wp: u32, odd: bool, q: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    if (QUANT_FORMAT == 2u) {
        // Q8_0 pair: even block scale = w0 low half, quants from byte 2; odd scale = w8
        // high half, quants w9 .. w16.
        if (odd) {
            let scale = unpack2x16float(weights[wp + 8u]).y;
            return scale * (dot(signed_bytes(weights[wp + 9u + 2u * q]), xa)
                + dot(signed_bytes(weights[wp + 10u + 2u * q]), xb));
        }
        let scale = unpack2x16float(weights[wp]).x;
        let w0 = weights[wp + 2u * q];
        let w1 = weights[wp + 2u * q + 1u];
        let w2 = weights[wp + 2u * q + 2u];
        return scale * (dot(signed_bytes((w0 >> 16u) | (w1 << 16u)), xa)
            + dot(signed_bytes((w1 >> 16u) | (w2 << 16u)), xb));
    }
    // Q4_0 pair: even block scale = w0 low half, quants from byte 2; odd scale = w4 high
    // half, quants w5 .. w8. Word q holds values 4q .. 4q + 3 (low) and 16 + 4q .. (high).
    if (odd) {
        let scale = unpack2x16float(weights[wp + 4u]).y;
        let w = weights[wp + 5u + q];
        return scale * (dot(low_nibbles(w), xa) + dot(high_nibbles(w), xb));
    }
    let scale = unpack2x16float(weights[wp]).x;
    let w = (weights[wp + q] >> 16u) | (weights[wp + q + 1u] << 16u);
    return scale * (dot(low_nibbles(w), xa) + dot(high_nibbles(w), xb));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>,
        @builtin(local_invocation_index) t: u32,
        @builtin(subgroup_id) wave: u32, @builtin(num_subgroups) waves: u32) {
    let tiles_n = (u.N + ROWS - 1u) / ROWS;
    let tile = group.x + group.y * grid.x;
    let m = tile / tiles_n;
    if (m >= u.M) { return; }
    let row0 = (tile % tiles_n) * ROWS;
    let half = WORKGROUP_SIZE / 2u;
    let odd = t >= half;
    let lane = t % half;
    let q = lane % 4u;
    let pairs = u.blocks / 2u;
    let words = select(9u, 17u, QUANT_FORMAT == 2u);
    let x_base = (u.a_offset + m * u.K) / 4u;
    var acc = vec4<f32>(0.0);
    for (var p = lane / 4u; p < pairs; p += half / 4u) {
        let block = 2u * p + select(0u, 1u, odd);
        var xa: vec4<f32>;
        var xb: vec4<f32>;
        if (QUANT_FORMAT == 2u) {
            xa = input[x_base + block * 8u + 2u * q];
            xb = input[x_base + block * 8u + 2u * q + 1u];
        } else {
            xa = input[x_base + block * 8u + q];
            xb = input[x_base + block * 8u + q + 4u];
        }
        for (var r = 0u; r < ROWS; r++) {
            let row = row0 + r;
            if (row < u.N) {
                acc[r] += quarter_dot(((u.b_row + row) * pairs + p) * words, odd, q, xa, xb);
            }
        }
    }
    let total = subgroupAdd(acc);
    if (subgroupElect()) { wave_sums[wave] = total; }
    workgroupBarrier();
    if (t < ROWS) {
        let row = row0 + t;
        var sum = 0.0;
        for (var w = 0u; w < waves; w++) { sum += wave_sums[w][t]; }
        if (row < u.N) { output[u.c_offset + m * u.N + row] = sum * u.alpha; }
    }
}
