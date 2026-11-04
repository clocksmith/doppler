// AUTO-GENERATED from src/gpu/kernels/matmul_q4_0_tile64.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
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
    if ((offset & 1u) == 0u) {
        let bits = (weights[offset / 4u] >> ((offset & 2u) * 8u)) & 0xffffu;
        return unpack2x16float(bits).x;
    }
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

// 128 weight rows x 128 token columns, 256 invocations. Staging reuses each
// decoded weight across 128 tokens. Per-output F32 FMA order remains K ascending.
// Native Q4_0 and Q4_1 only; registry admission requires 32768 shared bytes.
const TILE_ROWS: u32 = 128u;
const TILE_TOKENS: u32 = 128u;
const STAGE: u32 = 32u;
var<workgroup> w_tile: array<vec4<f32>, STAGE * 32u>;
var<workgroup> x_tile: array<vec4<f32>, STAGE * 32u>;

fn tile_word(offset: u32) -> u32 {
    let shift = (offset % 4u) * 8u;
    let low = weights[offset / 4u];
    if (shift == 0u) { return low; }
    return (low >> shift) | (weights[offset / 4u + 1u] << (32u - shift));
}
fn unpack_q4_0_lo(word: u32, scale: f32) -> vec4<f32> {
    return vec4<f32>(
        scale * f32(i32((word >> 0u) & 15u) - 8),
        scale * f32(i32((word >> 8u) & 15u) - 8),
        scale * f32(i32((word >> 16u) & 15u) - 8),
        scale * f32(i32((word >> 24u) & 15u) - 8)
    );
}
fn unpack_q4_0_hi(word: u32, scale: f32) -> vec4<f32> {
    return vec4<f32>(
        scale * f32(i32((word >> 4u) & 15u) - 8),
        scale * f32(i32((word >> 12u) & 15u) - 8),
        scale * f32(i32((word >> 20u) & 15u) - 8),
        scale * f32(i32((word >> 28u) & 15u) - 8)
    );
}
fn unpack_q4_1_lo(word: u32, scale: f32, min_val: f32) -> vec4<f32> {
    return vec4<f32>(
        scale * f32((word >> 0u) & 15u) + min_val,
        scale * f32((word >> 8u) & 15u) + min_val,
        scale * f32((word >> 16u) & 15u) + min_val,
        scale * f32((word >> 24u) & 15u) + min_val
    );
}
fn unpack_q4_1_hi(word: u32, scale: f32, min_val: f32) -> vec4<f32> {
    return vec4<f32>(
        scale * f32((word >> 4u) & 15u) + min_val,
        scale * f32((word >> 12u) & 15u) + min_val,
        scale * f32((word >> 20u) & 15u) + min_val,
        scale * f32((word >> 28u) & 15u) + min_val
    );
}
fn unpack_q8_0_word(word: u32, scale: f32) -> vec4<f32> {
    return vec4<f32>(
        scale * f32(bitcast<i32>(((word >> 0u) & 255u) << 24u) >> 24),
        scale * f32(bitcast<i32>(((word >> 8u) & 255u) << 24u) >> 24),
        scale * f32(bitcast<i32>(((word >> 16u) & 255u) << 24u) >> 24),
        scale * f32(bitcast<i32>(((word >> 24u) & 255u) << 24u) >> 24)
    );
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let tiles_n = (u.N + TILE_ROWS - 1u) / TILE_ROWS;
    let tiles_m = (u.M + TILE_TOKENS - 1u) / TILE_TOKENS;
    let tile = group.x + group.y * grid.x;
    let live_tile = tile < tiles_n * tiles_m;
    let n0 = (tile % tiles_n) * TILE_ROWS;
    let m0 = (tile / tiles_n) * TILE_TOKENS;
    let row_group = lane % 16u;
    let token_group = lane / 16u;
    var acc0_0 = vec4<f32>(0.0);
    var acc0_1 = vec4<f32>(0.0);
    var acc1_0 = vec4<f32>(0.0);
    var acc1_1 = vec4<f32>(0.0);
    var acc2_0 = vec4<f32>(0.0);
    var acc2_1 = vec4<f32>(0.0);
    var acc3_0 = vec4<f32>(0.0);
    var acc3_1 = vec4<f32>(0.0);
    var acc4_0 = vec4<f32>(0.0);
    var acc4_1 = vec4<f32>(0.0);
    var acc5_0 = vec4<f32>(0.0);
    var acc5_1 = vec4<f32>(0.0);
    var acc6_0 = vec4<f32>(0.0);
    var acc6_1 = vec4<f32>(0.0);
    var acc7_0 = vec4<f32>(0.0);
    var acc7_1 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < u.K; k0 += STAGE) {
        if (lane < 256u) {
            let xl = lane;
            let kq = xl % 8u;
            let cq = xl / 8u;
            var x0 = vec4<f32>(0.0);
            var x1 = vec4<f32>(0.0);
            var x2 = vec4<f32>(0.0);
            var x3 = vec4<f32>(0.0);
            if (live_tile) {
                let k_col = k0 + kq * 4u;
                let m_base = m0 + cq * 4u;
                if (m_base + 0u < u.M) {
                    let b0 = u.a_offset + (m_base + 0u) * u.K + k_col;
                    x0 = vec4<f32>(input[b0], input[b0 + 1u], input[b0 + 2u], input[b0 + 3u]);
                }
                if (m_base + 1u < u.M) {
                    let b1 = u.a_offset + (m_base + 1u) * u.K + k_col;
                    x1 = vec4<f32>(input[b1], input[b1 + 1u], input[b1 + 2u], input[b1 + 3u]);
                }
                if (m_base + 2u < u.M) {
                    let b2 = u.a_offset + (m_base + 2u) * u.K + k_col;
                    x2 = vec4<f32>(input[b2], input[b2 + 1u], input[b2 + 2u], input[b2 + 3u]);
                }
                if (m_base + 3u < u.M) {
                    let b3 = u.a_offset + (m_base + 3u) * u.K + k_col;
                    x3 = vec4<f32>(input[b3], input[b3 + 1u], input[b3 + 2u], input[b3 + 3u]);
                }
            }
            let k_base = kq * 4u;
            x_tile[(k_base + 0u) * 32u + cq] = vec4<f32>(x0.x, x1.x, x2.x, x3.x);
            x_tile[(k_base + 1u) * 32u + cq] = vec4<f32>(x0.y, x1.y, x2.y, x3.y);
            x_tile[(k_base + 2u) * 32u + cq] = vec4<f32>(x0.z, x1.z, x2.z, x3.z);
            x_tile[(k_base + 3u) * 32u + cq] = vec4<f32>(x0.w, x1.w, x2.w, x3.w);
        }
        if (QUANT_FORMAT == 0u || QUANT_FORMAT == 1u) {
            if (lane < 128u) {
                let rq = lane / 4u;
                let jj = lane % 4u;
                var lo: array<vec4<f32>, 4>;
                var hi: array<vec4<f32>, 4>;
                let block_idx = k0 / 32u;
                let data_offset = select(2u, 4u, QUANT_FORMAT == 1u);
                for (var r = 0u; r < 4u; r++) {
                    let w_row = n0 + rq * 4u + r;
                    if (live_tile && w_row < u.N) {
                        let base = ((u.b_row + w_row) * u.blocks + block_idx) * BLOCK_BYTES;
                        let scale = packed_half(base);
                        let word = tile_word(base + data_offset + jj * 4u);
                        if (QUANT_FORMAT == 1u) {
                            let min_val = packed_half(base + 2u);
                            lo[r] = unpack_q4_1_lo(word, scale, min_val);
                            hi[r] = unpack_q4_1_hi(word, scale, min_val);
                        }
                        else {
                            lo[r] = unpack_q4_0_lo(word, scale);
                            hi[r] = unpack_q4_0_hi(word, scale);
                        }
                    }
                    else {
                        lo[r] = vec4<f32>(0.0);
                        hi[r] = vec4<f32>(0.0);
                    }
                }
                let k_lo = jj * 4u;
                w_tile[(k_lo + 0u) * 32u + rq] = vec4<f32>(lo[0].x, lo[1].x, lo[2].x, lo[3].x);
                w_tile[(k_lo + 1u) * 32u + rq] = vec4<f32>(lo[0].y, lo[1].y, lo[2].y, lo[3].y);
                w_tile[(k_lo + 2u) * 32u + rq] = vec4<f32>(lo[0].z, lo[1].z, lo[2].z, lo[3].z);
                w_tile[(k_lo + 3u) * 32u + rq] = vec4<f32>(lo[0].w, lo[1].w, lo[2].w, lo[3].w);
                let k_hi = 16u + jj * 4u;
                w_tile[(k_hi + 0u) * 32u + rq] = vec4<f32>(hi[0].x, hi[1].x, hi[2].x, hi[3].x);
                w_tile[(k_hi + 1u) * 32u + rq] = vec4<f32>(hi[0].y, hi[1].y, hi[2].y, hi[3].y);
                w_tile[(k_hi + 2u) * 32u + rq] = vec4<f32>(hi[0].z, hi[1].z, hi[2].z, hi[3].z);
                w_tile[(k_hi + 3u) * 32u + rq] = vec4<f32>(hi[0].w, hi[1].w, hi[2].w, hi[3].w);
            }
        }
        workgroupBarrier();
        for (var k = 0u; k < STAGE; k++) {
            let w0 = w_tile[k * 32u + row_group * 2u + 0u];
            let w1 = w_tile[k * 32u + row_group * 2u + 1u];
            let x0 = x_tile[k * 32u + token_group * 2u + 0u];
            let x1 = x_tile[k * 32u + token_group * 2u + 1u];
            acc0_0 = fma(w0, vec4<f32>(x0.x), acc0_0);
            acc0_1 = fma(w1, vec4<f32>(x0.x), acc0_1);
            acc1_0 = fma(w0, vec4<f32>(x0.y), acc1_0);
            acc1_1 = fma(w1, vec4<f32>(x0.y), acc1_1);
            acc2_0 = fma(w0, vec4<f32>(x0.z), acc2_0);
            acc2_1 = fma(w1, vec4<f32>(x0.z), acc2_1);
            acc3_0 = fma(w0, vec4<f32>(x0.w), acc3_0);
            acc3_1 = fma(w1, vec4<f32>(x0.w), acc3_1);
            acc4_0 = fma(w0, vec4<f32>(x1.x), acc4_0);
            acc4_1 = fma(w1, vec4<f32>(x1.x), acc4_1);
            acc5_0 = fma(w0, vec4<f32>(x1.y), acc5_0);
            acc5_1 = fma(w1, vec4<f32>(x1.y), acc5_1);
            acc6_0 = fma(w0, vec4<f32>(x1.z), acc6_0);
            acc6_1 = fma(w1, vec4<f32>(x1.z), acc6_1);
            acc7_0 = fma(w0, vec4<f32>(x1.w), acc7_0);
            acc7_1 = fma(w1, vec4<f32>(x1.w), acc7_1);
        }
        workgroupBarrier();
    }
    if (!live_tile) {
        return;
    }
    if (m0 + token_group * 8u + 0u < u.M) {
        if (n0 + row_group * 8u + 0u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 0u) * u.N + n0 + row_group * 8u + 0u] = acc0_0.x * u.alpha;
        }
        if (n0 + row_group * 8u + 1u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 0u) * u.N + n0 + row_group * 8u + 1u] = acc0_0.y * u.alpha;
        }
        if (n0 + row_group * 8u + 2u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 0u) * u.N + n0 + row_group * 8u + 2u] = acc0_0.z * u.alpha;
        }
        if (n0 + row_group * 8u + 3u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 0u) * u.N + n0 + row_group * 8u + 3u] = acc0_0.w * u.alpha;
        }
        if (n0 + row_group * 8u + 4u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 0u) * u.N + n0 + row_group * 8u + 4u] = acc0_1.x * u.alpha;
        }
        if (n0 + row_group * 8u + 5u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 0u) * u.N + n0 + row_group * 8u + 5u] = acc0_1.y * u.alpha;
        }
        if (n0 + row_group * 8u + 6u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 0u) * u.N + n0 + row_group * 8u + 6u] = acc0_1.z * u.alpha;
        }
        if (n0 + row_group * 8u + 7u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 0u) * u.N + n0 + row_group * 8u + 7u] = acc0_1.w * u.alpha;
        }
    }
    if (m0 + token_group * 8u + 1u < u.M) {
        if (n0 + row_group * 8u + 0u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 1u) * u.N + n0 + row_group * 8u + 0u] = acc1_0.x * u.alpha;
        }
        if (n0 + row_group * 8u + 1u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 1u) * u.N + n0 + row_group * 8u + 1u] = acc1_0.y * u.alpha;
        }
        if (n0 + row_group * 8u + 2u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 1u) * u.N + n0 + row_group * 8u + 2u] = acc1_0.z * u.alpha;
        }
        if (n0 + row_group * 8u + 3u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 1u) * u.N + n0 + row_group * 8u + 3u] = acc1_0.w * u.alpha;
        }
        if (n0 + row_group * 8u + 4u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 1u) * u.N + n0 + row_group * 8u + 4u] = acc1_1.x * u.alpha;
        }
        if (n0 + row_group * 8u + 5u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 1u) * u.N + n0 + row_group * 8u + 5u] = acc1_1.y * u.alpha;
        }
        if (n0 + row_group * 8u + 6u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 1u) * u.N + n0 + row_group * 8u + 6u] = acc1_1.z * u.alpha;
        }
        if (n0 + row_group * 8u + 7u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 1u) * u.N + n0 + row_group * 8u + 7u] = acc1_1.w * u.alpha;
        }
    }
    if (m0 + token_group * 8u + 2u < u.M) {
        if (n0 + row_group * 8u + 0u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 2u) * u.N + n0 + row_group * 8u + 0u] = acc2_0.x * u.alpha;
        }
        if (n0 + row_group * 8u + 1u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 2u) * u.N + n0 + row_group * 8u + 1u] = acc2_0.y * u.alpha;
        }
        if (n0 + row_group * 8u + 2u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 2u) * u.N + n0 + row_group * 8u + 2u] = acc2_0.z * u.alpha;
        }
        if (n0 + row_group * 8u + 3u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 2u) * u.N + n0 + row_group * 8u + 3u] = acc2_0.w * u.alpha;
        }
        if (n0 + row_group * 8u + 4u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 2u) * u.N + n0 + row_group * 8u + 4u] = acc2_1.x * u.alpha;
        }
        if (n0 + row_group * 8u + 5u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 2u) * u.N + n0 + row_group * 8u + 5u] = acc2_1.y * u.alpha;
        }
        if (n0 + row_group * 8u + 6u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 2u) * u.N + n0 + row_group * 8u + 6u] = acc2_1.z * u.alpha;
        }
        if (n0 + row_group * 8u + 7u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 2u) * u.N + n0 + row_group * 8u + 7u] = acc2_1.w * u.alpha;
        }
    }
    if (m0 + token_group * 8u + 3u < u.M) {
        if (n0 + row_group * 8u + 0u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 3u) * u.N + n0 + row_group * 8u + 0u] = acc3_0.x * u.alpha;
        }
        if (n0 + row_group * 8u + 1u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 3u) * u.N + n0 + row_group * 8u + 1u] = acc3_0.y * u.alpha;
        }
        if (n0 + row_group * 8u + 2u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 3u) * u.N + n0 + row_group * 8u + 2u] = acc3_0.z * u.alpha;
        }
        if (n0 + row_group * 8u + 3u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 3u) * u.N + n0 + row_group * 8u + 3u] = acc3_0.w * u.alpha;
        }
        if (n0 + row_group * 8u + 4u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 3u) * u.N + n0 + row_group * 8u + 4u] = acc3_1.x * u.alpha;
        }
        if (n0 + row_group * 8u + 5u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 3u) * u.N + n0 + row_group * 8u + 5u] = acc3_1.y * u.alpha;
        }
        if (n0 + row_group * 8u + 6u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 3u) * u.N + n0 + row_group * 8u + 6u] = acc3_1.z * u.alpha;
        }
        if (n0 + row_group * 8u + 7u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 3u) * u.N + n0 + row_group * 8u + 7u] = acc3_1.w * u.alpha;
        }
    }
    if (m0 + token_group * 8u + 4u < u.M) {
        if (n0 + row_group * 8u + 0u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 4u) * u.N + n0 + row_group * 8u + 0u] = acc4_0.x * u.alpha;
        }
        if (n0 + row_group * 8u + 1u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 4u) * u.N + n0 + row_group * 8u + 1u] = acc4_0.y * u.alpha;
        }
        if (n0 + row_group * 8u + 2u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 4u) * u.N + n0 + row_group * 8u + 2u] = acc4_0.z * u.alpha;
        }
        if (n0 + row_group * 8u + 3u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 4u) * u.N + n0 + row_group * 8u + 3u] = acc4_0.w * u.alpha;
        }
        if (n0 + row_group * 8u + 4u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 4u) * u.N + n0 + row_group * 8u + 4u] = acc4_1.x * u.alpha;
        }
        if (n0 + row_group * 8u + 5u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 4u) * u.N + n0 + row_group * 8u + 5u] = acc4_1.y * u.alpha;
        }
        if (n0 + row_group * 8u + 6u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 4u) * u.N + n0 + row_group * 8u + 6u] = acc4_1.z * u.alpha;
        }
        if (n0 + row_group * 8u + 7u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 4u) * u.N + n0 + row_group * 8u + 7u] = acc4_1.w * u.alpha;
        }
    }
    if (m0 + token_group * 8u + 5u < u.M) {
        if (n0 + row_group * 8u + 0u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 5u) * u.N + n0 + row_group * 8u + 0u] = acc5_0.x * u.alpha;
        }
        if (n0 + row_group * 8u + 1u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 5u) * u.N + n0 + row_group * 8u + 1u] = acc5_0.y * u.alpha;
        }
        if (n0 + row_group * 8u + 2u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 5u) * u.N + n0 + row_group * 8u + 2u] = acc5_0.z * u.alpha;
        }
        if (n0 + row_group * 8u + 3u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 5u) * u.N + n0 + row_group * 8u + 3u] = acc5_0.w * u.alpha;
        }
        if (n0 + row_group * 8u + 4u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 5u) * u.N + n0 + row_group * 8u + 4u] = acc5_1.x * u.alpha;
        }
        if (n0 + row_group * 8u + 5u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 5u) * u.N + n0 + row_group * 8u + 5u] = acc5_1.y * u.alpha;
        }
        if (n0 + row_group * 8u + 6u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 5u) * u.N + n0 + row_group * 8u + 6u] = acc5_1.z * u.alpha;
        }
        if (n0 + row_group * 8u + 7u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 5u) * u.N + n0 + row_group * 8u + 7u] = acc5_1.w * u.alpha;
        }
    }
    if (m0 + token_group * 8u + 6u < u.M) {
        if (n0 + row_group * 8u + 0u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 6u) * u.N + n0 + row_group * 8u + 0u] = acc6_0.x * u.alpha;
        }
        if (n0 + row_group * 8u + 1u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 6u) * u.N + n0 + row_group * 8u + 1u] = acc6_0.y * u.alpha;
        }
        if (n0 + row_group * 8u + 2u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 6u) * u.N + n0 + row_group * 8u + 2u] = acc6_0.z * u.alpha;
        }
        if (n0 + row_group * 8u + 3u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 6u) * u.N + n0 + row_group * 8u + 3u] = acc6_0.w * u.alpha;
        }
        if (n0 + row_group * 8u + 4u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 6u) * u.N + n0 + row_group * 8u + 4u] = acc6_1.x * u.alpha;
        }
        if (n0 + row_group * 8u + 5u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 6u) * u.N + n0 + row_group * 8u + 5u] = acc6_1.y * u.alpha;
        }
        if (n0 + row_group * 8u + 6u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 6u) * u.N + n0 + row_group * 8u + 6u] = acc6_1.z * u.alpha;
        }
        if (n0 + row_group * 8u + 7u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 6u) * u.N + n0 + row_group * 8u + 7u] = acc6_1.w * u.alpha;
        }
    }
    if (m0 + token_group * 8u + 7u < u.M) {
        if (n0 + row_group * 8u + 0u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 7u) * u.N + n0 + row_group * 8u + 0u] = acc7_0.x * u.alpha;
        }
        if (n0 + row_group * 8u + 1u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 7u) * u.N + n0 + row_group * 8u + 1u] = acc7_0.y * u.alpha;
        }
        if (n0 + row_group * 8u + 2u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 7u) * u.N + n0 + row_group * 8u + 2u] = acc7_0.z * u.alpha;
        }
        if (n0 + row_group * 8u + 3u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 7u) * u.N + n0 + row_group * 8u + 3u] = acc7_0.w * u.alpha;
        }
        if (n0 + row_group * 8u + 4u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 7u) * u.N + n0 + row_group * 8u + 4u] = acc7_1.x * u.alpha;
        }
        if (n0 + row_group * 8u + 5u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 7u) * u.N + n0 + row_group * 8u + 5u] = acc7_1.y * u.alpha;
        }
        if (n0 + row_group * 8u + 6u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 7u) * u.N + n0 + row_group * 8u + 6u] = acc7_1.z * u.alpha;
        }
        if (n0 + row_group * 8u + 7u < u.N) {
            output[u.c_offset + (m0 + token_group * 8u + 7u) * u.N + n0 + row_group * 8u + 7u] = acc7_1.w * u.alpha;
        }
    }
}
