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
// Original interleaved GGUF blocks. Named words keep unaligned native loads
// and unpacking in registers instead of dynamically indexing a private array.
struct NativeBlock {
    scale: f32, minimum: f32,
    q0: u32, q1: u32, q2: u32, q3: u32,
    q4: u32, q5: u32, q6: u32, q7: u32,
}
fn aligned_quants(low: u32, high: u32, unaligned: bool) -> u32 {
    if (unaligned) { return high; }
    return (low >> 16u) | (high << 16u);
}
fn load_native_block(base: u32) -> NativeBlock {
    let first = base / 4u;
    let unaligned = (base % 4u) != 0u;
    let b0 = bank_word(first);
    let b1 = bank_word(first + 1u);
    let b2 = bank_word(first + 2u);
    let b3 = bank_word(first + 3u);
    let b4 = bank_word(first + 4u);
    var block: NativeBlock;
    block.scale = unpack2x16float(b0 >> select(0u, 16u, unaligned)).x;
    if (QUANT_FORMAT == 1u) {
        block.minimum = unpack2x16float(b0).y;
        block.q0 = b1; block.q1 = b2; block.q2 = b3; block.q3 = b4;
    } else {
        block.q0 = aligned_quants(b0, b1, unaligned);
        block.q1 = aligned_quants(b1, b2, unaligned);
        block.q2 = aligned_quants(b2, b3, unaligned);
        block.q3 = aligned_quants(b3, b4, unaligned);
    }
    if (QUANT_FORMAT == 2u) {
        let b5 = bank_word(first + 5u);
        let b6 = bank_word(first + 6u);
        let b7 = bank_word(first + 7u);
        let b8 = bank_word(first + 8u);
        block.q4 = aligned_quants(b4, b5, unaligned);
        block.q5 = aligned_quants(b5, b6, unaligned);
        block.q6 = aligned_quants(b6, b7, unaligned);
        block.q7 = aligned_quants(b7, b8, unaligned);
    }
    return block;
}
fn weight4(word: u32, high: bool, scale: f32, minimum: f32) -> vec4<f32> {
    if (QUANT_FORMAT == 2u) { return scale * signed4(word); }
    let q = (bytes4(word) >> vec4<u32>(select(0u, 4u, high))) & vec4<u32>(15u);
    if (QUANT_FORMAT == 1u) { return scale * vec4<f32>(q) + vec4<f32>(minimum); }
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
// Keep each pair's dot sequence unchanged while the caller names its accumulator.
fn pair_dot(w0: vec4<f32>, w1: vec4<f32>, w2: vec4<f32>, w3: vec4<f32>,
    w4: vec4<f32>, w5: vec4<f32>, w6: vec4<f32>, w7: vec4<f32>, p: u32, lane: u32) -> f32 {
    var sum = 0.0;
    sum += dot(w0, tile[p * 64u + 0u + lane]);
    sum += dot(w1, tile[p * 64u + 8u + lane]);
    sum += dot(w2, tile[p * 64u + 16u + lane]);
    sum += dot(w3, tile[p * 64u + 24u + lane]);
    sum += dot(w4, tile[p * 64u + 32u + lane]);
    sum += dot(w5, tile[p * 64u + 40u + lane]);
    sum += dot(w6, tile[p * 64u + 48u + lane]);
    sum += dot(w7, tile[p * 64u + 56u + lane]);
    return sum;
}
fn reduce_pair(p: u32, t: u32, row0: u32, acc: vec4<f32>) {
    workgroupBarrier();
    for (var r = 0u; r < ROWS_PER_THREAD; r++) {
        partial[r * WORKGROUP_SIZE + t] = acc[r];
    }
    workgroupBarrier();
    if (t < 32u * ROWS_PER_THREAD && row0 + t < u.N) {
        let r = t / 32u; let row_lane = t % 32u;
        var sum = 0.0;
        for (var l = 0u; l < 8u; l++) { sum += partial[r * WORKGROUP_SIZE + row_lane * 8u + l]; }
        output[route_ids[p] * u.N + row0 + t] = sum;
    }
}
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
    // Eight admitted route slots, with scalar-addressable row components.
    var acc0: vec4<f32>;
    var acc1: vec4<f32>;
    var acc2: vec4<f32>;
    var acc3: vec4<f32>;
    var acc4: vec4<f32>;
    var acc5: vec4<f32>;
    var acc6: vec4<f32>;
    var acc7: vec4<f32>;
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
                    let block = load_native_block(base);
                    let w0 = weight4(block.q0, false, block.scale, block.minimum);
                    let w1 = weight4(block.q1, false, block.scale, block.minimum);
                    let w2 = weight4(block.q2, false, block.scale, block.minimum);
                    let w3 = weight4(block.q3, false, block.scale, block.minimum);
                    let w4 = weight4(select(block.q0, block.q4, QUANT_FORMAT == 2u), true, block.scale, block.minimum);
                    let w5 = weight4(select(block.q1, block.q5, QUANT_FORMAT == 2u), true, block.scale, block.minimum);
                    let w6 = weight4(select(block.q2, block.q6, QUANT_FORMAT == 2u), true, block.scale, block.minimum);
                    let w7 = weight4(select(block.q3, block.q7, QUANT_FORMAT == 2u), true, block.scale, block.minimum);
                    if (n > 0u) { acc0[r] += pair_dot(w0, w1, w2, w3, w4, w5, w6, w7, 0u, lane); }
                    if (n > 1u) { acc1[r] += pair_dot(w0, w1, w2, w3, w4, w5, w6, w7, 1u, lane); }
                    if (n > 2u) { acc2[r] += pair_dot(w0, w1, w2, w3, w4, w5, w6, w7, 2u, lane); }
                    if (n > 3u) { acc3[r] += pair_dot(w0, w1, w2, w3, w4, w5, w6, w7, 3u, lane); }
                    if (n > 4u) { acc4[r] += pair_dot(w0, w1, w2, w3, w4, w5, w6, w7, 4u, lane); }
                    if (n > 5u) { acc5[r] += pair_dot(w0, w1, w2, w3, w4, w5, w6, w7, 5u, lane); }
                    if (n > 6u) { acc6[r] += pair_dot(w0, w1, w2, w3, w4, w5, w6, w7, 6u, lane); }
                    if (n > 7u) { acc7[r] += pair_dot(w0, w1, w2, w3, w4, w5, w6, w7, 7u, lane); }
                }
            }
        }
    }
    if (n > 0u) { reduce_pair(0u, t, row0, acc0); }
    if (n > 1u) { reduce_pair(1u, t, row0, acc1); }
    if (n > 2u) { reduce_pair(2u, t, row0, acc2); }
    if (n > 3u) { reduce_pair(3u, t, row0, acc3); }
    if (n > 4u) { reduce_pair(4u, t, row0, acc4); }
    if (n > 5u) { reduce_pair(5u, t, row0, acc5); }
    if (n > 6u) { reduce_pair(6u, t, row0, acc6); }
    if (n > 7u) { reduce_pair(7u, t, row0, acc7); }
}
