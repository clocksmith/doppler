// AUTO-GENERATED from src/gpu/kernels/matmul_q5k.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// Native source-preserving Q5_K projections.
// Projection topology follows Doppler native GGUF GEMV and tiled prefill.
// Native GGUF blocks remain interleaved; no Pooled split-scale repacking.
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
override COLS: u32 = 1u;
override BLOCK_VALUES: u32 = 256u;
override BLOCK_BYTES: u32 = 176u;
override QUANT_FORMAT: u32 = 5u;
struct Uniforms { M: u32, N: u32, K: u32, alpha: f32, blocks: u32, a_offset: u32, b_row: u32, c_offset: u32, }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> wave_sums: array<vec4<f32>, WORKGROUP_SIZE * COLS>;
fn q5k_scale_min(group: u32, first: u32, second: u32, third: u32) -> vec2<u32> {
    let shift = (group % 4u) * 8u;
    let a = (first >> shift) & 255u;
    let b = (second >> shift) & 255u;
    let c = (third >> shift) & 255u;
    if (group < 4u) { return vec2<u32>(a & 63u, b & 63u); }
    return vec2<u32>((c & 15u) | ((a >> 6u) << 4u),
        (c >> 4u) | ((b >> 6u) << 4u));
}
struct Quarter { first: vec4<f32>, second: vec4<f32> }
fn q5k_quarter(base: u32, group: u32, quarter: u32) -> Quarter {
    let word = base / 4u;
    let scales = unpack2x16float(weights[word]);
    let sm = q5k_scale_min(group, weights[word + 1u], weights[word + 2u], weights[word + 3u]);
    let d = scales.x * f32(sm.x);
    let minimum = scales.y * f32(sm.y);
    let low_word = word + 12u + (group / 2u) * 8u + quarter;
    let shift = (group % 2u) * 4u;
    let high0 = vec4<u32>(unpack4xU8(weights[word + 4u + quarter]));
    let high1 = vec4<u32>(unpack4xU8(weights[word + 8u + quarter]));
    let low0 = vec4<u32>(unpack4xU8((weights[low_word] >> shift) & 0x0F0F0F0Fu));
    let low1 = vec4<u32>(unpack4xU8((weights[low_word + 4u] >> shift) & 0x0F0F0F0Fu));
    let q0 = low0 | (((high0 >> vec4<u32>(group)) & vec4<u32>(1u)) << vec4<u32>(4u));
    let q1 = low1 | (((high1 >> vec4<u32>(group)) & vec4<u32>(1u)) << vec4<u32>(4u));
    return Quarter(d * vec4<f32>(q0) - vec4<f32>(minimum), d * vec4<f32>(q1) - vec4<f32>(minimum));
}
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>,
        @builtin(local_invocation_index) t: u32, @builtin(subgroup_id) wave: u32, @builtin(num_subgroups) waves: u32) {
    let tiles_n = (u.N + ROWS - 1u) / ROWS;
    let tile = group.x + group.y * grid.x;
    let first_m = (tile / tiles_n) * COLS;
    if (first_m >= u.M) { return; }
    let row0 = (tile % tiles_n) * ROWS;
    let g = (t % 32u) / 4u;
    let q = t % 4u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var block = t / 32u; block < u.blocks; block += WORKGROUP_SIZE / 32u) {
        let x = block * 64u + g * 8u + q;
        var xa0 = vec4<f32>(0.0); var xb0 = vec4<f32>(0.0);
        if (0u < COLS && first_m + 0u < u.M) {
            let at = (u.a_offset + (first_m + 0u) * u.K) / 4u + x;
            xa0 = input[at]; xb0 = input[at + 4u];
        }
        var xa1 = vec4<f32>(0.0); var xb1 = vec4<f32>(0.0);
        if (1u < COLS && first_m + 1u < u.M) {
            let at = (u.a_offset + (first_m + 1u) * u.K) / 4u + x;
            xa1 = input[at]; xb1 = input[at + 4u];
        }
        var xa2 = vec4<f32>(0.0); var xb2 = vec4<f32>(0.0);
        if (2u < COLS && first_m + 2u < u.M) {
            let at = (u.a_offset + (first_m + 2u) * u.K) / 4u + x;
            xa2 = input[at]; xb2 = input[at + 4u];
        }
        var xa3 = vec4<f32>(0.0); var xb3 = vec4<f32>(0.0);
        if (3u < COLS && first_m + 3u < u.M) {
            let at = (u.a_offset + (first_m + 3u) * u.K) / 4u + x;
            xa3 = input[at]; xb3 = input[at + 4u];
        }
        for (var r = 0u; r < ROWS; r++) {
            let row = row0 + r;
            if (row < u.N) {
                let decoded = q5k_quarter(((u.b_row + row) * u.blocks + block) * BLOCK_BYTES, g, q);
                if (0u < COLS && first_m + 0u < u.M) { acc0[r] += dot(decoded.first, xa0) + dot(decoded.second, xb0); }
                if (1u < COLS && first_m + 1u < u.M) { acc1[r] += dot(decoded.first, xa1) + dot(decoded.second, xb1); }
                if (2u < COLS && first_m + 2u < u.M) { acc2[r] += dot(decoded.first, xa2) + dot(decoded.second, xb2); }
                if (3u < COLS && first_m + 3u < u.M) { acc3[r] += dot(decoded.first, xa3) + dot(decoded.second, xb3); }
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
