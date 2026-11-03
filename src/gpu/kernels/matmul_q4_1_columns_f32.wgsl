// AUTO-GENERATED from src/gpu/kernels/matmul_gguf_pair_f32.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// Native GGUF Q4_1: one aligned scale/min word and four quant words per block.
// Decode a quarter once and reuse its F32 values across activation columns.
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
override ROWS: u32 = 2u;
override COLS: u32 = 4u;
override QUANT_FORMAT: u32 = 1u;
override BLOCK_VALUES: u32 = 32u;
override BLOCK_BYTES: u32 = 20u;

struct Uniforms {
    M: u32, N: u32, K: u32, alpha: f32,
    blocks: u32, a_offset: u32, b_row: u32, c_offset: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> wave_sums: array<vec4<f32>, WORKGROUP_SIZE * COLS>;


// Decode each weight quarter once before applying it to all activation columns.
struct BlockQuarter { scale: f32, first: vec4<f32>, second: vec4<f32> }
fn decode_quarter(wp: u32, q: u32) -> BlockQuarter {
    let w = weights[wp + 1u + q];
    let scales=unpack2x16float(weights[wp]);
    return BlockQuarter(1.0, scales.x * vec4<f32>(unpack4xU8(w & 0x0F0F0F0Fu)) + vec4<f32>(scales.y),
      scales.x * vec4<f32>(unpack4xU8((w >> 4u) & 0x0F0F0F0Fu)) + vec4<f32>(scales.y));
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
    let half = WORKGROUP_SIZE;
    let lane = t;
    let q = lane % 4u;
    let pairs = u.blocks;
    let words = 5u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var p = lane / 4u; p < pairs; p += half / 4u) {
        let block = p;
        for (var r = 0u; r < ROWS; r++) {
            let row = row0 + r;
            if (row < u.N) {
                let wp = ((u.b_row + row) * pairs + p) * words;
                let decoded = decode_quarter(wp, q);
                if (0u < COLS && first_m + 0u < u.M) {
                    let x_base = (u.a_offset + (first_m + 0u) * u.K) / 4u;
                    var xa: vec4<f32>;
                    var xb: vec4<f32>;
                    xa = input[x_base + block * 8u + q];
                    xb = input[x_base + block * 8u + q + 4u];
                    acc0[r] += decoded.scale * (dot(decoded.first, xa) + dot(decoded.second, xb));
                }
                if (1u < COLS && first_m + 1u < u.M) {
                    let x_base = (u.a_offset + (first_m + 1u) * u.K) / 4u;
                    var xa: vec4<f32>;
                    var xb: vec4<f32>;
                    xa = input[x_base + block * 8u + q];
                    xb = input[x_base + block * 8u + q + 4u];
                    acc1[r] += decoded.scale * (dot(decoded.first, xa) + dot(decoded.second, xb));
                }
                if (2u < COLS && first_m + 2u < u.M) {
                    let x_base = (u.a_offset + (first_m + 2u) * u.K) / 4u;
                    var xa: vec4<f32>;
                    var xb: vec4<f32>;
                    xa = input[x_base + block * 8u + q];
                    xb = input[x_base + block * 8u + q + 4u];
                    acc2[r] += decoded.scale * (dot(decoded.first, xa) + dot(decoded.second, xb));
                }
                if (3u < COLS && first_m + 3u < u.M) {
                    let x_base = (u.a_offset + (first_m + 3u) * u.K) / 4u;
                    var xa: vec4<f32>;
                    var xb: vec4<f32>;
                    xa = input[x_base + block * 8u + q];
                    xb = input[x_base + block * 8u + q + 4u];
                    acc3[r] += decoded.scale * (dot(decoded.first, xa) + dot(decoded.second, xb));
                }
            }
        }
    }
    if (0u < COLS) { let total = subgroupAdd(acc0); if (subgroupElect()) { wave_sums[0u * WORKGROUP_SIZE + wave] = total; } }
    if (1u < COLS) { let total = subgroupAdd(acc1); if (subgroupElect()) { wave_sums[1u * WORKGROUP_SIZE + wave] = total; } }
    if (2u < COLS) { let total = subgroupAdd(acc2); if (subgroupElect()) { wave_sums[2u * WORKGROUP_SIZE + wave] = total; } }
    if (3u < COLS) { let total = subgroupAdd(acc3); if (subgroupElect()) { wave_sums[3u * WORKGROUP_SIZE + wave] = total; } }
    workgroupBarrier();
    if (t < ROWS) {
        let row = row0 + t;
        for (var c = 0u; c < COLS; c++) {
            let m = first_m + c;
            var sum = 0.0;
            for (var w = 0u; w < waves; w++) { sum += wave_sums[c * WORKGROUP_SIZE + w][t]; }
            if (m < u.M && row < u.N) { output[u.c_offset + m * u.N + row] = sum * u.alpha; }
        }
    }
}
