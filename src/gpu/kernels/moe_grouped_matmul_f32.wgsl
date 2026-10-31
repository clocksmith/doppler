// Adapted from Pooled engine/wgsl/moe_group.js, revision
// 6b4a24f73cce7ef9a9e3ab35b9a0d4c09b51dfff: stable counting sort and F32 row tiles.
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
// 
override WORKGROUP_SIZE: u32;
override GROUP_SIZE: u32;
override ROWS_PER_THREAD: u32;
override INPUT_IS_TOKEN: bool;
override SPLIT_GATE_UP: bool;
override QUANT_FORMAT: u32;
override BLOCK_BYTES: u32;
struct Uniforms {
    num_routes: u32, num_experts: u32, top_k: u32, pair_offset: u32,
    N: u32, K: u32, _pad0: u32, _pad1: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read> up_weights: array<u32>;
@group(0) @binding(4) var<storage, read> groups: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
var<private> use_up: bool;
fn bank_word(i: u32) -> u32 {
    if (use_up) { return up_weights[i]; }
    return weights[i];
}
fn bytes4(word: u32) -> vec4<u32> {
    return (vec4<u32>(word) >> vec4<u32>(0u,8u,16u,24u)) & vec4<u32>(255u);
}
fn signed4(word: u32) -> vec4<f32> {
    let v = bitcast<i32>(word);
    return vec4<f32>(f32((v << 24u) >> 24u), f32((v << 16u) >> 24u),
        f32((v << 8u) >> 24u), f32(v >> 24u));
}
// Original interleaved GGUF blocks, each read once: the word-aligned span holding
// the block's scale (and Q4_1 minimum) and quants, 5 words for Q4_0 / Q4_1 and 9
// for Q8_0. Q4_1's signed minimum is applied per value; Q4_0's low/high nibbles
// describe the first/second sixteen values.
var<private> block_words: array<u32, 9>;
var<private> block_shift: u32;
fn load_block(base: u32) {
    let first = base / 4u;
    let words = select(5u, 9u, QUANT_FORMAT == 2u);
    for (var i = 0u; i < words; i++) { block_words[i] = bank_word(first + i); }
    block_shift = (base % 4u) * 8u;
}
// The block word at byte `offset` from the block start (offset a multiple of 2).
fn block_word(offset: u32) -> u32 {
    let at = offset * 8u + block_shift;
    let low = block_words[at / 32u];
    if (at % 32u == 0u) { return low; }
    return (low >> (at % 32u)) | (block_words[at / 32u + 1u] << (32u - at % 32u));
}
fn block_half(offset: u32) -> f32 {
    let at = offset * 8u + block_shift;
    return unpack2x16float(block_words[at / 32u] >> (at % 32u)).x;
}
// Values 4j .. 4j + 3 of the loaded block.
fn weight4(j: u32) -> vec4<f32> {
    let scale = block_half(0u);
    if (QUANT_FORMAT == 2u) { return scale * signed4(block_word(2u + j * 4u)); }
    let data_offset = select(2u, 4u, QUANT_FORMAT == 1u);
    let q = (bytes4(block_word(data_offset + (j % 4u) * 4u))
        >> vec4<u32>(select(0u,4u,j >= 4u))) & vec4<u32>(15u);
    if (QUANT_FORMAT == 1u) { return scale * vec4<f32>(q) + vec4<f32>(block_half(2u)); }
    return scale * (vec4<f32>(q) - vec4<f32>(8.0));
}
// Whole-vector stores only (overlapping component writes are unsafe on Metal). tile
// holds, per pair, the 8 blocks x 8 four-value groups of the current K tile:
// tile[pair * 64 + j * 8 + b] = input values 4j .. 4j + 3 of block b, the values
// weight4(j) multiplies, so lane b reads one vec4 per j, conflict free.
var<workgroup> tile: array<vec4<f32>, 512>;
var<workgroup> partial: array<f32, 1024>;
var<workgroup> route_ids: array<u32, 8>;
var<workgroup> chunk_count: u32;
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    if (t == 0u) { chunk_count = groups[wg.y * 4u + 1u]; }
    let n = workgroupUniformLoad(&chunk_count);
    let first = groups[wg.y * 4u];
    let expert = groups[wg.y * 4u + 2u];
    if (t < n) { route_ids[t] = groups[u.pair_offset + first + t]; }
    let lane = t & 7u;
    let row0 = wg.x * (32u * ROWS_PER_THREAD);
    let blocks = u.K / 32u;
    var acc: array<f32, 32>;
    for (var kt = 0u; kt < blocks; kt += 8u) {
        workgroupBarrier();
        for (var i = t; i < GROUP_SIZE * 64u; i += WORKGROUP_SIZE) {
            let pair = i / 64u;
            let j = (i / 8u) % 8u;
            let b = i % 8u;
            if (pair < n && kt + b < blocks) {
                var input_row = route_ids[pair];
                if (INPUT_IS_TOKEN) { input_row /= u.top_k; }
                let k = input_row * u.K + (kt + b) * 32u + j * 4u;
                tile[i] = vec4<f32>(input[k], input[k + 1u], input[k + 2u], input[k + 3u]);
            }
        }
        workgroupBarrier();
        let block_idx = kt + lane;
        if (block_idx < blocks) {
            for (var r = 0u; r < ROWS_PER_THREAD; r++) {
                let row = row0 + r * 32u + (t >> 3u);
                if (row < u.N) {
                    var bank_rows = u.N;
                    var bank_row = row;
                    use_up = false;
                    if (SPLIT_GATE_UP) {
                        bank_rows /= 2u;
                        use_up = row >= bank_rows;
                        bank_row = row % bank_rows;
                    }
                    // This lane's block of the row, dequantized once and reused by every
                    // pair: per (pair, row) the same sum in the same order.
                    let base = ((expert * bank_rows + bank_row) * blocks + block_idx) * BLOCK_BYTES;
                    load_block(base);
                    var w: array<vec4<f32>, 8>;
                    for (var j = 0u; j < 8u; j++) { w[j] = weight4(j); }
                    for (var p = 0u; p < n; p++) {
                        var sum = 0.0;
                        for (var j = 0u; j < 8u; j++) { sum += dot(w[j], tile[p * 64u + j * 8u + lane]); }
                        acc[p * ROWS_PER_THREAD + r] += sum;
                    }
                }
            }
        }
    }
    for (var p = 0u; p < n; p++) {
        workgroupBarrier();
        for (var r = 0u; r < ROWS_PER_THREAD; r++) {
            partial[r * WORKGROUP_SIZE + t] = acc[p * ROWS_PER_THREAD + r];
        }
        workgroupBarrier();
        if (t < 32u * ROWS_PER_THREAD && row0 + t < u.N) {
            let r = t / 32u; let row_lane = t % 32u;
            var sum = 0.0;
            for (var l = 0u; l < 8u; l++) { sum += partial[r * WORKGROUP_SIZE + row_lane * 8u + l]; }
            output[route_ids[p] * u.N + row0 + t] = sum;
        }
    }
}
