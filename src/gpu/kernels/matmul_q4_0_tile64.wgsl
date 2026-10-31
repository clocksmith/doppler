// Tiled native GGUF block GEMM for prefill (M > 1). Bindings, uniforms and the block decoder
// follow matmul_q4_0.wgsl; the attribution below covers the shared decoder.
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
override WORKGROUP_SIZE: u32 = 128u;
override COLS_PER_WG: u32 = 4u;
override THREADS_PER_COL_GEMV: u32 = 32u;
override TILE_SIZE: u32 = 16u;
struct Uniforms {
    M: u32, N: u32, K: u32, alpha: f32,
    blocks: u32, a_offset: u32, b_row: u32, c_offset: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
override BLOCK_VALUES: u32 = 32u;
override BLOCK_BYTES: u32 = 18u;
override QUANT_FORMAT: u32 = 0u;
fn q40_byte(offset: u32) -> u32 {
    return (weights[offset / 4u] >> ((offset % 4u) * 8u)) & 255u;
}
fn packed_half(offset: u32) -> f32 {
    return unpack2x16float(q40_byte(offset) | (q40_byte(offset + 1u) << 8u)).x;
}
fn packed_signed_byte(offset: u32) -> i32 {
    return bitcast<i32>(q40_byte(offset) << 24u) >> 24;
}
fn q40_value(row: u32, k: u32, blocks: u32) -> f32 {
    let offset = (row * blocks + k / BLOCK_VALUES) * BLOCK_BYTES;
    let local = k % BLOCK_VALUES;
    if (QUANT_FORMAT == 3u) {
        let section = (local % 128u) / 32u;
        let lane = local % 32u;
        let low = q40_byte(offset + (local / 128u) * 64u + (section % 2u) * 32u + lane);
        let high = q40_byte(offset + 128u + (local / 128u) * 32u + lane);
        let value = ((low >> ((section / 2u) * 4u)) & 15u) | (((high >> (section * 2u)) & 3u) << 4u);
        return packed_half(offset + 208u) * f32(packed_signed_byte(offset + 192u + local / 16u)) * f32(i32(value) - 32);
    }
    let scale = packed_half(offset);
    if (QUANT_FORMAT == 2u) { return scale * f32(packed_signed_byte(offset + 2u + local)); }
    if (QUANT_FORMAT == 4u) {
        // Q5_0: f16 scale, 32 high bits (one per value), 16 bytes of low nibbles.
        let high_bits = q40_byte(offset + 2u) | (q40_byte(offset + 3u) << 8u)
            | (q40_byte(offset + 4u) << 16u) | (q40_byte(offset + 5u) << 24u);
        let low = (q40_byte(offset + 6u + local % 16u) >> select(0u, 4u, local >= 16u)) & 15u;
        return scale * f32(i32(low | (((high_bits >> local) & 1u) << 4u)) - 16);
    }
    let data_offset = select(2u,4u,QUANT_FORMAT == 1u);
    let packed = q40_byte(offset + data_offset + local % 16u);
    let shift = select(0u,4u,local >= 16u);
    let value = (packed >> shift) & 15u;
    if (QUANT_FORMAT == 1u) { return scale * f32(value) + packed_half(offset + 2u); }
    return scale * f32(i32(value) - 8);
}

// 64 weight rows x 64 token columns per workgroup, 256 invocations, each owning
// 4 rows x 4 tokens in registers. Per 32-value K stage the workgroup dequantizes
// the weight tile once into workgroup memory (k-major, vec4 over 4 rows) and
// stages the activation tile (k-major, vec4 over 4 tokens). One F32 accumulator
// per output, k ascending, no split-K: deterministic and tile-position independent.
// Same tile shape as Pooled engine/wgsl/gemm_wide.js WIDE_TILE_DEFAULT (6b4a24f7),
// reading native GGUF blocks instead of Pooled's repacked layout.
const TILE_ROWS: u32 = 64u;
const TILE_TOKENS: u32 = 64u;
const STAGE: u32 = 32u;
var<workgroup> w_tile: array<vec4<f32>, STAGE * 16u>;
var<workgroup> x_tile: array<vec4<f32>, STAGE * 16u>;

fn tile_word(offset: u32) -> u32 {
    let shift = (offset % 4u) * 8u;
    let low = weights[offset / 4u];
    if (shift == 0u) { return low; }
    return (low >> shift) | (weights[offset / 4u + 1u] << (32u - shift));
}
// Eight consecutive values (k0 + quarter * 8 ...) of one row. Q4_0 and Q8_0 read
// two packed words; other formats use the scalar decoder. Values are identical.
fn tile_values(row: u32, k: u32, quarter: u32) -> array<f32, 8> {
    var out: array<f32, 8>;
    if (QUANT_FORMAT == 0u || QUANT_FORMAT == 2u) {
        let base = (row * u.blocks + k / 32u) * BLOCK_BYTES;
        let scale = packed_half(base);
        if (QUANT_FORMAT == 2u) {
            let a = tile_word(base + 2u + quarter * 8u);
            let b = tile_word(base + 6u + quarter * 8u);
            for (var i = 0u; i < 4u; i++) {
                out[i] = scale * f32(bitcast<i32>(((a >> (i * 8u)) & 255u) << 24u) >> 24);
                out[i + 4u] = scale * f32(bitcast<i32>(((b >> (i * 8u)) & 255u) << 24u) >> 24);
            }
        } else {
            let first = base + 2u + (quarter % 2u) * 8u;
            let shift = (quarter / 2u) * 4u;
            let a = tile_word(first);
            let b = tile_word(first + 4u);
            for (var i = 0u; i < 4u; i++) {
                out[i] = scale * f32(i32((a >> (i * 8u + shift)) & 15u) - 8);
                out[i + 4u] = scale * f32(i32((b >> (i * 8u + shift)) & 15u) - 8);
            }
        }
        return out;
    }
    for (var i = 0u; i < 8u; i++) { out[i] = q40_value(row, k + quarter * 8u + i, u.blocks); }
    return out;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let tiles_n = (u.N + TILE_ROWS - 1u) / TILE_ROWS;
    let tiles_m = (u.M + TILE_TOKENS - 1u) / TILE_TOKENS;
    let tile = group.x + group.y * grid.x;
    let live_tile = tile < tiles_n * tiles_m;
    let n0 = (tile % tiles_n) * TILE_ROWS;
    let m0 = (tile / tiles_n) * TILE_TOKENS;
    let row_group = lane % 16u;
    let token_group = lane / 16u;
    // Loader roles: 64 rows (or tokens) x 4 quarters of 8 values.
    let load_item = lane / 4u;
    let load_quarter = lane % 4u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < u.K; k0 += STAGE) {
        let w_row = n0 + load_item;
        let x_token = m0 + load_item;
        var wv: array<f32, 8>;
        if (live_tile && w_row < u.N) { wv = tile_values(u.b_row + w_row, k0, load_quarter); }
        for (var i = 0u; i < 8u; i++) {
            let k = load_quarter * 8u + i;
            w_tile[k * 16u + load_item / 4u][load_item % 4u] = wv[i];
            var x = 0.0;
            if (live_tile && x_token < u.M) { x = input[u.a_offset + x_token * u.K + k0 + k]; }
            x_tile[k * 16u + load_item / 4u][load_item % 4u] = x;
        }
        workgroupBarrier();
        for (var k = 0u; k < STAGE; k++) {
            let w = w_tile[k * 16u + row_group];
            let x = x_tile[k * 16u + token_group];
            acc0 = fma(w, vec4<f32>(x.x), acc0);
            acc1 = fma(w, vec4<f32>(x.y), acc1);
            acc2 = fma(w, vec4<f32>(x.z), acc2);
            acc3 = fma(w, vec4<f32>(x.w), acc3);
        }
        workgroupBarrier();
    }
    if (!live_tile) { return; }
    let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
    for (var j = 0u; j < 4u; j++) {
        let m = m0 + token_group * 4u + j;
        if (m >= u.M) { continue; }
        for (var i = 0u; i < 4u; i++) {
            let n = n0 + row_group * 4u + i;
            if (n < u.N) { output[u.c_offset + m * u.N + n] = accs[j][i] * u.alpha; }
        }
    }
}
