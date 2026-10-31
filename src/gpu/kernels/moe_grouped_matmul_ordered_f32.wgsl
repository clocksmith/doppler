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
// Grouped weight reuse with the scalar FMA sequence and shared reduction tree
// of moe_route_gguf_split_f32 / moe_route_q4_0_f32 (32 lanes per output).
// Kept separate from the block-dot candidate to preserve its source identity.
override WORKGROUP_SIZE: u32;
override GROUP_SIZE: u32;
override ROWS_PER_THREAD: u32;
override REDUCTION_LANES: u32;
override INPUT_IS_TOKEN: bool;
override SPLIT_GATE_UP: bool;
override QUANT_FORMAT: u32;
override BLOCK_BYTES: u32;

const BLOCK_VALUES: u32 = 32u;
const TILE_BLOCKS: u32 = 8u;
const TILE_VALUES: u32 = BLOCK_VALUES * TILE_BLOCKS;
const MAX_GROUP_SIZE: u32 = 8u;
const MAX_ROWS_PER_THREAD: u32 = 4u;
const MAX_WORKGROUP_SIZE: u32 = 256u;

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
fn bank_byte(offset: u32) -> u32 {
    var word: u32;
    if (use_up) { word = up_weights[offset / 4u]; }
    else { word = weights[offset / 4u]; }
    return (word >> ((offset % 4u) * 8u)) & 255u;
}
fn packed_half(offset: u32) -> f32 {
    return unpack2x16float(bank_byte(offset) | (bank_byte(offset + 1u) << 8u)).x;
}
fn weight_value(offset: u32, local: u32) -> f32 {
    let scale = packed_half(offset);
    if (QUANT_FORMAT == 2u) {
        let signed = bitcast<i32>(bank_byte(offset + 2u + local) << 24u) >> 24;
        return scale * f32(signed);
    }
    let data_offset = select(2u, 4u, QUANT_FORMAT == 1u);
    let packed = bank_byte(offset + data_offset + local % 16u);
    let shift = select(0u, 4u, local >= 16u);
    let value = (packed >> shift) & 15u;
    if (QUANT_FORMAT == 1u) { return scale * f32(value) + packed_half(offset + 2u); }
    return scale * f32(i32(value) - 8);
}

var<workgroup> tile: array<f32, MAX_GROUP_SIZE * TILE_VALUES>;
var<workgroup> partial: array<f32, MAX_ROWS_PER_THREAD * MAX_WORKGROUP_SIZE>;
var<workgroup> route_ids: array<u32, MAX_GROUP_SIZE>;
var<workgroup> chunk_count: u32;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    if (t == 0u) { chunk_count = groups[wg.y * 4u + 1u]; }
    let n = workgroupUniformLoad(&chunk_count);
    let first = groups[wg.y * 4u];
    let expert = groups[wg.y * 4u + 2u];
    if (t < n) { route_ids[t] = groups[u.pair_offset + first + t]; }
    let lane = t % REDUCTION_LANES;
    let row_groups = WORKGROUP_SIZE / REDUCTION_LANES;
    let row0 = wg.x * (row_groups * ROWS_PER_THREAD);
    let blocks = u.K / BLOCK_VALUES;
    var acc: array<f32, MAX_GROUP_SIZE * MAX_ROWS_PER_THREAD>;

    for (var kt = 0u; kt < blocks; kt += TILE_BLOCKS) {
        workgroupBarrier();
        for (var i = t; i < GROUP_SIZE * TILE_VALUES; i += WORKGROUP_SIZE) {
            let pair = i / TILE_VALUES;
            let value = i % TILE_VALUES;
            let k = kt * BLOCK_VALUES + value;
            if (pair < n && k < u.K) {
                var input_row = route_ids[pair];
                if (INPUT_IS_TOKEN) { input_row /= u.top_k; }
                tile[pair * TILE_VALUES + (value % BLOCK_VALUES) * TILE_BLOCKS
                    + value / BLOCK_VALUES] = input[input_row * u.K + k];
            }
        }
        workgroupBarrier();
        for (var r = 0u; r < ROWS_PER_THREAD; r++) {
            let row = row0 + r * row_groups + t / REDUCTION_LANES;
            if (row < u.N) {
                var bank_rows = u.N;
                var bank_row = row;
                use_up = false;
                if (SPLIT_GATE_UP) {
                    bank_rows /= 2u;
                    use_up = row >= bank_rows;
                    bank_row = row % bank_rows;
                }
                for (var b = 0u; b < TILE_BLOCKS && kt + b < blocks; b++) {
                    let base = ((expert * bank_rows + bank_row) * blocks + kt + b) * BLOCK_BYTES;
                    let weight = weight_value(base, lane);
                    for (var p = 0u; p < n; p++) {
                        let at = p * TILE_VALUES + lane * TILE_BLOCKS + b;
                        let a = p * ROWS_PER_THREAD + r;
                        // No intermediate block sum: lane k = lane + 32*b
                        // stays in ascending order across every staging tile.
                        acc[a] = fma(tile[at], weight, acc[a]);
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
        for (var stride = REDUCTION_LANES / 2u; stride > 0u; stride /= 2u) {
            if (lane < stride) {
                for (var r = 0u; r < ROWS_PER_THREAD; r++) {
                    let at = r * WORKGROUP_SIZE + t;
                    partial[at] += partial[at + stride];
                }
            }
            workgroupBarrier();
        }
        if (lane == 0u) {
            for (var r = 0u; r < ROWS_PER_THREAD; r++) {
                let row = row0 + r * row_groups + t / REDUCTION_LANES;
                if (row < u.N) { output[route_ids[p] * u.N + row] = partial[r * WORKGROUP_SIZE + t]; }
            }
        }
    }
}
