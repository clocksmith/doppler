// AUTO-GENERATED from src/gpu/kernels/matmul_gguf_pair_f32.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// Hoisted four-column activations; bounded wave-batch reduction scratch.
// Native GGUF Q4_0 / Q8_0 (main) and Q6_K (main_q6k) GEMV over aligned block pairs. Two consecutive blocks of a
// row with an even block count start on a 4-byte boundary (Q4_0 9 words, Q8_0 17),
// and the odd block's scale and quants are word-aligned. Invocations [0, WG/2) take
// even blocks and [WG/2, WG) odd blocks, so every wave runs one decode path; four
// invocations share a block (quarters, as in the cooperative GEMV) and each reuses its
// activation slice across ROWS rows. Waves reduce with subgroupAdd; bounded wave batches fold into slots,
// which are added in slot order. The stored bytes are the GGUF bytes.
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
override COLS: u32 = 4u;
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

const WAVE_SLOTS: u32 = 16u;
var<workgroup> wave_sums: array<vec4<f32>, WAVE_SLOTS * COLS>;

fn low_nibbles(w: u32) -> vec4<f32> { return vec4<f32>(unpack4xU8(w & 0x0F0F0F0Fu)) - vec4<f32>(8.0); }
fn high_nibbles(w: u32) -> vec4<f32> { return vec4<f32>(unpack4xU8((w >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0); }
fn signed_bytes(w: u32) -> vec4<f32> { return vec4<f32>(unpack4xI8(w)); }

// Decode each weight quarter once before applying it to all activation columns.
struct BlockQuarter { scale: f32, first: vec4<f32>, second: vec4<f32> }
fn decode_quarter(wp: u32, odd: bool, q: u32) -> BlockQuarter {
    if (QUANT_FORMAT == 2u) {
        if (odd) {
            return BlockQuarter(unpack2x16float(weights[wp + 8u]).y,
                signed_bytes(weights[wp + 9u + 2u * q]), signed_bytes(weights[wp + 10u + 2u * q]));
        }
        let w0 = weights[wp + 2u * q];
        let w1 = weights[wp + 2u * q + 1u];
        let w2 = weights[wp + 2u * q + 2u];
        return BlockQuarter(unpack2x16float(weights[wp]).x,
            signed_bytes((w0 >> 16u) | (w1 << 16u)), signed_bytes((w1 >> 16u) | (w2 << 16u)));
    }
    if (odd) {
        let w = weights[wp + 5u + q];
        return BlockQuarter(unpack2x16float(weights[wp + 4u]).y, low_nibbles(w), high_nibbles(w));
    }
    let w = (weights[wp + q] >> 16u) | (weights[wp + q + 1u] << 16u);
    return BlockQuarter(unpack2x16float(weights[wp]).x, low_nibbles(w), high_nibbles(w));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>,
        @builtin(local_invocation_index) t: u32,
        @builtin(subgroup_id) wave: u32, @builtin(num_subgroups) waves: u32) {
    let tiles_n = (u.N + ROWS - 1u) / ROWS;
    let tile = group.x + group.y * grid.x;
    let first_m = (tile / tiles_n) * COLS;
    if (first_m >= u.M) { return; }
    let row0 = (tile % tiles_n) * ROWS;
    let half = WORKGROUP_SIZE / 2u;
    let odd = t >= half;
    let lane = t % half;
    let q = lane % 4u;
    let pairs = u.blocks / 2u;
    let words = select(9u, 17u, QUANT_FORMAT == 2u);
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var p = lane / 4u; p < pairs; p += half / 4u) {
        let block = 2u * p + select(0u, 1u, odd);
        var xa0 = vec4<f32>(0.0); var xb0 = vec4<f32>(0.0);
        if (0u < COLS && first_m + 0u < u.M) {
            let x_base = (u.a_offset + (first_m + 0u) * u.K) / 4u;
            if (QUANT_FORMAT == 2u) {
                xa0 = input[x_base + block * 8u + 2u * q];
                xb0 = input[x_base + block * 8u + 2u * q + 1u];
            } else {
                xa0 = input[x_base + block * 8u + q];
                xb0 = input[x_base + block * 8u + q + 4u];
            }
        }
        var xa1 = vec4<f32>(0.0); var xb1 = vec4<f32>(0.0);
        if (1u < COLS && first_m + 1u < u.M) {
            let x_base = (u.a_offset + (first_m + 1u) * u.K) / 4u;
            if (QUANT_FORMAT == 2u) {
                xa1 = input[x_base + block * 8u + 2u * q];
                xb1 = input[x_base + block * 8u + 2u * q + 1u];
            } else {
                xa1 = input[x_base + block * 8u + q];
                xb1 = input[x_base + block * 8u + q + 4u];
            }
        }
        var xa2 = vec4<f32>(0.0); var xb2 = vec4<f32>(0.0);
        if (2u < COLS && first_m + 2u < u.M) {
            let x_base = (u.a_offset + (first_m + 2u) * u.K) / 4u;
            if (QUANT_FORMAT == 2u) {
                xa2 = input[x_base + block * 8u + 2u * q];
                xb2 = input[x_base + block * 8u + 2u * q + 1u];
            } else {
                xa2 = input[x_base + block * 8u + q];
                xb2 = input[x_base + block * 8u + q + 4u];
            }
        }
        var xa3 = vec4<f32>(0.0); var xb3 = vec4<f32>(0.0);
        if (3u < COLS && first_m + 3u < u.M) {
            let x_base = (u.a_offset + (first_m + 3u) * u.K) / 4u;
            if (QUANT_FORMAT == 2u) {
                xa3 = input[x_base + block * 8u + 2u * q];
                xb3 = input[x_base + block * 8u + 2u * q + 1u];
            } else {
                xa3 = input[x_base + block * 8u + q];
                xb3 = input[x_base + block * 8u + q + 4u];
            }
        }
        for (var r = 0u; r < ROWS; r++) {
            let row = row0 + r;
            if (row < u.N) {
                let wp = ((u.b_row + row) * pairs + p) * words;
                let decoded = decode_quarter(wp, odd, q);
                if (0u < COLS && first_m + 0u < u.M) {
                    acc0[r] += decoded.scale * (dot(decoded.first, xa0) + dot(decoded.second, xb0));
                }
                if (1u < COLS && first_m + 1u < u.M) {
                    acc1[r] += decoded.scale * (dot(decoded.first, xa1) + dot(decoded.second, xb1));
                }
                if (2u < COLS && first_m + 2u < u.M) {
                    acc2[r] += decoded.scale * (dot(decoded.first, xa2) + dot(decoded.second, xb2));
                }
                if (3u < COLS && first_m + 3u < u.M) {
                    acc3[r] += decoded.scale * (dot(decoded.first, xa3) + dot(decoded.second, xb3));
                }
            }
        }
    }
    let total0 = subgroupAdd(acc0);
    let total1 = subgroupAdd(acc1);
    let total2 = subgroupAdd(acc2);
    let total3 = subgroupAdd(acc3);
    // Fold wave batches safely; no minimum subgroup width is assumed.
    for (var batch = 0u; batch < (waves + WAVE_SLOTS - 1u) / WAVE_SLOTS; batch++) {
        if (subgroupElect() && wave / WAVE_SLOTS == batch) {
            let slot = wave % WAVE_SLOTS;
            if (0u < COLS) {
                if (batch == 0u) { wave_sums[0u * WAVE_SLOTS + slot] = total0; }
                else { wave_sums[0u * WAVE_SLOTS + slot] += total0; }
            }
            if (1u < COLS) {
                if (batch == 0u) { wave_sums[1u * WAVE_SLOTS + slot] = total1; }
                else { wave_sums[1u * WAVE_SLOTS + slot] += total1; }
            }
            if (2u < COLS) {
                if (batch == 0u) { wave_sums[2u * WAVE_SLOTS + slot] = total2; }
                else { wave_sums[2u * WAVE_SLOTS + slot] += total2; }
            }
            if (3u < COLS) {
                if (batch == 0u) { wave_sums[3u * WAVE_SLOTS + slot] = total3; }
                else { wave_sums[3u * WAVE_SLOTS + slot] += total3; }
            }
        }
        workgroupBarrier();
    }
    if (t < ROWS) {
        let row = row0 + t;
        for (var c = 0u; c < COLS; c++) {
            let m = first_m + c;
            var sum = 0.0;
            for (var w = 0u; w < min(waves, WAVE_SLOTS); w++) { sum += wave_sums[c * WAVE_SLOTS + w][t]; }
            if (m < u.M && row < u.N) { output[u.c_offset + m * u.N + row] = sum * u.alpha; }
        }
    }
}

// Q6_K (main_q6k): super-block pairs (420 bytes, 105 words). Even super-blocks start
// word-aligned, odd ones two bytes into a word; workgroup halves take even and odd
// super-blocks. 32 invocations per super-block, each one ql word j of half n: low
// nibbles are values 128n + j .. + 3, high nibbles 128n + j + 64 .. + 3, with their
// two high bits from qh, one int8 scale per 16 values and the f16 super-block scale.

// The word at byte offset `byte` (even) of the weights.
fn word_at(byte: u32) -> u32 {
    let w = byte / 4u;
    if ((byte & 2u) == 0u) { return weights[w]; }
    return (weights[w] >> 16u) | (weights[w + 1u] << 16u);
}

fn signed_byte_at(byte: u32) -> f32 {
    return f32(bitcast<i32>(((weights[byte / 4u] >> ((byte % 4u) * 8u)) & 255u) << 24u) >> 24u);
}

struct Q6Quarter { scale: f32, firstScale: f32, secondScale: f32, first: vec4<f32>, second: vec4<f32>, value: u32 }
fn decode_q6k_quarter(base: u32, s: u32) -> Q6Quarter {
    let n = s / 16u;
    let j = (s % 16u) * 4u;
    let ql = word_at(base + 64u * n + j);
    let qh = vec4<u32>(unpack4xU8(word_at(base + 128u + 32u * n + j % 32u)));
    let shift = select(0u, 2u, j >= 32u);
    let lo = vec4<u32>(unpack4xU8(ql & 0x0F0F0F0Fu)) | (((qh >> vec4<u32>(shift)) & vec4<u32>(3u)) << vec4<u32>(4u));
    let hi = vec4<u32>(unpack4xU8((ql >> 4u) & 0x0F0F0F0Fu)) | (((qh >> vec4<u32>(shift + 4u)) & vec4<u32>(3u)) << vec4<u32>(4u));
    let d = unpack2x16float(word_at(base + 208u)).x;
    let value = 128u * n + j;
    return Q6Quarter(d, signed_byte_at(base + 192u + value / 16u),
        signed_byte_at(base + 192u + (value + 64u) / 16u),
        vec4<f32>(lo) - vec4<f32>(32.0), vec4<f32>(hi) - vec4<f32>(32.0), value);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main_q6k(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>,
            @builtin(local_invocation_index) t: u32,
            @builtin(subgroup_id) wave: u32, @builtin(num_subgroups) waves: u32) {
    let tiles_n = (u.N + ROWS - 1u) / ROWS;
    let tile = group.x + group.y * grid.x;
    let first_m = (tile / tiles_n) * COLS;
    if (first_m >= u.M) { return; }
    let row0 = (tile % tiles_n) * ROWS;
    let half = WORKGROUP_SIZE / 2u;
    let odd = t >= half;
    let lane = t % half;
    let s = lane % 32u;
    let pairs = u.blocks / 2u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var p = lane / 32u; p < pairs; p += half / 32u) {
        let block = 2u * p + select(0u, 1u, odd);
        let input_value = 128u * (s / 16u) + (s % 16u) * 4u;
        var xa0 = vec4<f32>(0.0); var xb0 = vec4<f32>(0.0);
        if (0u < COLS && first_m + 0u < u.M) {
            let x_base = (u.a_offset + (first_m + 0u) * u.K) / 4u;
            xa0 = input[x_base + block * 64u + input_value / 4u];
            xb0 = input[x_base + block * 64u + (input_value + 64u) / 4u];
        }
        var xa1 = vec4<f32>(0.0); var xb1 = vec4<f32>(0.0);
        if (1u < COLS && first_m + 1u < u.M) {
            let x_base = (u.a_offset + (first_m + 1u) * u.K) / 4u;
            xa1 = input[x_base + block * 64u + input_value / 4u];
            xb1 = input[x_base + block * 64u + (input_value + 64u) / 4u];
        }
        var xa2 = vec4<f32>(0.0); var xb2 = vec4<f32>(0.0);
        if (2u < COLS && first_m + 2u < u.M) {
            let x_base = (u.a_offset + (first_m + 2u) * u.K) / 4u;
            xa2 = input[x_base + block * 64u + input_value / 4u];
            xb2 = input[x_base + block * 64u + (input_value + 64u) / 4u];
        }
        var xa3 = vec4<f32>(0.0); var xb3 = vec4<f32>(0.0);
        if (3u < COLS && first_m + 3u < u.M) {
            let x_base = (u.a_offset + (first_m + 3u) * u.K) / 4u;
            xa3 = input[x_base + block * 64u + input_value / 4u];
            xb3 = input[x_base + block * 64u + (input_value + 64u) / 4u];
        }
        for (var r = 0u; r < ROWS; r++) {
            let row = row0 + r;
            if (row < u.N) {
                let decoded = decode_q6k_quarter(((u.b_row + row) * u.blocks + block) * BLOCK_BYTES, s);
                if (0u < COLS && first_m + 0u < u.M) {
                    acc0[r] += decoded.scale * (decoded.firstScale * dot(decoded.first, xa0)
                        + decoded.secondScale * dot(decoded.second, xb0));
                }
                if (1u < COLS && first_m + 1u < u.M) {
                    acc1[r] += decoded.scale * (decoded.firstScale * dot(decoded.first, xa1)
                        + decoded.secondScale * dot(decoded.second, xb1));
                }
                if (2u < COLS && first_m + 2u < u.M) {
                    acc2[r] += decoded.scale * (decoded.firstScale * dot(decoded.first, xa2)
                        + decoded.secondScale * dot(decoded.second, xb2));
                }
                if (3u < COLS && first_m + 3u < u.M) {
                    acc3[r] += decoded.scale * (decoded.firstScale * dot(decoded.first, xa3)
                        + decoded.secondScale * dot(decoded.second, xb3));
                }
            }
        }
    }
    let total0 = subgroupAdd(acc0);
    let total1 = subgroupAdd(acc1);
    let total2 = subgroupAdd(acc2);
    let total3 = subgroupAdd(acc3);
    // Fold wave batches safely; no minimum subgroup width is assumed.
    for (var batch = 0u; batch < (waves + WAVE_SLOTS - 1u) / WAVE_SLOTS; batch++) {
        if (subgroupElect() && wave / WAVE_SLOTS == batch) {
            let slot = wave % WAVE_SLOTS;
            if (0u < COLS) {
                if (batch == 0u) { wave_sums[0u * WAVE_SLOTS + slot] = total0; }
                else { wave_sums[0u * WAVE_SLOTS + slot] += total0; }
            }
            if (1u < COLS) {
                if (batch == 0u) { wave_sums[1u * WAVE_SLOTS + slot] = total1; }
                else { wave_sums[1u * WAVE_SLOTS + slot] += total1; }
            }
            if (2u < COLS) {
                if (batch == 0u) { wave_sums[2u * WAVE_SLOTS + slot] = total2; }
                else { wave_sums[2u * WAVE_SLOTS + slot] += total2; }
            }
            if (3u < COLS) {
                if (batch == 0u) { wave_sums[3u * WAVE_SLOTS + slot] = total3; }
                else { wave_sums[3u * WAVE_SLOTS + slot] += total3; }
            }
        }
        workgroupBarrier();
    }
    if (t < ROWS) {
        let row = row0 + t;
        for (var c = 0u; c < COLS; c++) {
            let m = first_m + c;
            var sum = 0.0;
            for (var w = 0u; w < min(waves, WAVE_SLOTS); w++) { sum += wave_sums[c * WAVE_SLOTS + w][t]; }
            if (m < u.M && row < u.N) { output[u.c_offset + m * u.N + row] = sum * u.alpha; }
        }
    }
}
