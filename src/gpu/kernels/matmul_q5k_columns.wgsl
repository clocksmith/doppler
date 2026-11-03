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
override BATCH_ROWS: u32 = 4u;
override WORKGROUP_SIZE: u32 = 128u;
override COLS_PER_WG: u32 = 4u;
override THREADS_PER_COL_GEMV: u32 = 32u;
override TILE_SIZE: u32 = 16u;
override BLOCK_VALUES: u32 = 256u;
override BLOCK_BYTES: u32 = 176u;
override QUANT_FORMAT: u32 = 5u;
struct Uniforms {
    M: u32, N: u32, K: u32, alpha: f32,
    blocks: u32, a_offset: u32, b_row: u32, c_offset: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
fn q5k_byte(offset: u32) -> u32 {
    return (weights[offset / 4u] >> ((offset % 4u) * 8u)) & 255u;
}
fn q5k_half(offset: u32) -> f32 {
    return unpack2x16float(q5k_byte(offset) | (q5k_byte(offset + 1u) << 8u)).x;
}
// GGML Q5_K: two f16 block scales, twelve packed scale/min bytes,
// thirty-two high-bit bytes and 128 low-nibble bytes for 256 values.
// Reference: ggml-org/llama.cpp ggml/src/ggml-quants.c dequantize_row_q5_K.
fn q5k_value(row: u32, k: u32, blocks: u32) -> f32 {
    let base = (row * blocks + k / BLOCK_VALUES) * BLOCK_BYTES;
    let local = k % BLOCK_VALUES;
    let group = local / 32u;
    var scale: u32;
    var minimum: u32;
    if (group < 4u) {
        scale = q5k_byte(base + 4u + group) & 63u;
        minimum = q5k_byte(base + 8u + group) & 63u;
    } else {
        scale = (q5k_byte(base + 8u + group) & 15u)
            | ((q5k_byte(base + group) >> 6u) << 4u);
        minimum = (q5k_byte(base + 8u + group) >> 4u)
            | ((q5k_byte(base + 4u + group) >> 6u) << 4u);
    }
    let low = (q5k_byte(base + 48u + (local / 64u) * 32u + local % 32u)
        >> ((group % 2u) * 4u)) & 15u;
    let high = (q5k_byte(base + 16u + local % 32u) >> group) & 1u;
    let d = q5k_half(base) * f32(scale);
    let m = q5k_half(base + 2u) * f32(minimum);
    return d * f32(low | (high << 4u)) - m;
}
var<workgroup> partial: array<f32, WORKGROUP_SIZE>;
var<workgroup> tile_a: array<f32, TILE_SIZE * TILE_SIZE>;
var<workgroup> tile_b: array<f32, TILE_SIZE * TILE_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let tiles_n = (u.N + COLS_PER_WG - 1u) / COLS_PER_WG;
    let tile = group.x + group.y * grid.x;
    let m = tile / tiles_n;
    if (m >= u.M) { return; }
    let column = lane / THREADS_PER_COL_GEMV;
    let thread = lane % THREADS_PER_COL_GEMV;
    let n = (tile % tiles_n) * COLS_PER_WG + column;
    var sum = 0.0f;
    if (n < u.N) {
        for (var k = thread; k < u.K; k += THREADS_PER_COL_GEMV) {
            sum = fma(input[u.a_offset + m * u.K + k], q5k_value(u.b_row + n, k, u.blocks), sum);
        }
    }
    partial[lane] = sum;
    workgroupBarrier();
    for (var stride = THREADS_PER_COL_GEMV / 2u; stride > 0u; stride /= 2u) {
        if (thread < stride) { partial[lane] += partial[lane + stride]; }
        workgroupBarrier();
    }
    if (thread == 0u && n < u.N) { output[u.c_offset + m * u.N + n] = partial[lane] * u.alpha; }
}

// Keep the baseline reduction order while reading each block header and high
// bit byte once per lane. The registered variant uses 32 lanes for each native
// Q5_K group of 32 values; loop topology is block, group, then the same reduction.
fn q5k_scale_min(group: u32, first: u32, second: u32, third: u32) -> vec2<u32> {
    let shift = (group % 4u) * 8u;
    let a = (first >> shift) & 255u;
    let b = (second >> shift) & 255u;
    let c = (third >> shift) & 255u;
    if (group < 4u) { return vec2<u32>(a & 63u, b & 63u); }
    return vec2<u32>((c & 15u) | ((a >> 6u) << 4u),
        (c >> 4u) | ((b >> 6u) << 4u));
}

var<workgroup> partial_batch: array<vec4<f32>, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn gemv_blockwise(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let tiles_n = (u.N + COLS_PER_WG - 1u) / COLS_PER_WG;
    let tile = group.x + group.y * grid.x;
    let first_m = (tile / tiles_n) * BATCH_ROWS;
    if (first_m >= u.M) { return; }
    let column = lane / THREADS_PER_COL_GEMV;
    let thread = lane % THREADS_PER_COL_GEMV;
    let n = (tile % tiles_n) * COLS_PER_WG + column;
    var sum = vec4<f32>(0.0f);
    if (n < u.N) {
        for (var block = 0u; block < u.blocks; block++) {
            let base = ((u.b_row + n) * u.blocks + block) * BLOCK_BYTES;
            let scales = unpack2x16float(weights[base / 4u]);
            let first = weights[base / 4u + 1u];
            let second = weights[base / 4u + 2u];
            let third = weights[base / 4u + 3u];
            let high = q5k_byte(base + 16u + thread);
            let low = vec4<u32>(q5k_byte(base + 48u + thread),
                q5k_byte(base + 80u + thread), q5k_byte(base + 112u + thread),
                q5k_byte(base + 144u + thread));
            for (var index = 0u; index < 8u; index++) {
                let scale_min = q5k_scale_min(index, first, second, third);
                let quant = ((low[index / 2u] >> ((index % 2u) * 4u)) & 15u)
                    | (((high >> index) & 1u) << 4u);
                let d = scales.x * f32(scale_min.x);
                let minimum = scales.y * f32(scale_min.y);
                let value = d * f32(quant) - minimum;
                let k = block * BLOCK_VALUES + index * 32u + thread;
                for (var batch = 0u; batch < BATCH_ROWS; batch++) {
                    let m = first_m + batch;
                    if (m < u.M) { sum[batch] = fma(input[u.a_offset + m * u.K + k], value, sum[batch]); }
                }
            }
        }
    }
    partial_batch[lane] = sum;
    workgroupBarrier();
    for (var stride = THREADS_PER_COL_GEMV / 2u; stride > 0u; stride /= 2u) {
        if (thread < stride) { partial_batch[lane] += partial_batch[lane + stride]; }
        workgroupBarrier();
    }
    if (thread == 0u && n < u.N) {
        for (var batch = 0u; batch < BATCH_ROWS; batch++) {
            let m = first_m + batch;
            if (m < u.M) { output[u.c_offset + m * u.N + n] = partial_batch[lane][batch] * u.alpha; }
        }
    }
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn prefill(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>,
           @builtin(local_invocation_id) lane: vec3<u32>) {
    let tiles_n = (u.N + TILE_SIZE - 1u) / TILE_SIZE;
    let tile = group.x + group.y * grid.x;
    let tiles_m = (u.M + TILE_SIZE - 1u) / TILE_SIZE;
    if (tile >= tiles_m * tiles_n) { return; }
    let m = (tile / tiles_n) * TILE_SIZE + lane.y;
    let n = (tile % tiles_n) * TILE_SIZE + lane.x;
    let index = lane.y * TILE_SIZE + lane.x;
    var sum = 0.0f;
    for (var base = 0u; base < u.K; base += TILE_SIZE) {
        var a = 0.0f;
        var b = 0.0f;
        if (m < u.M && base + lane.x < u.K) { a = input[u.a_offset + m * u.K + base + lane.x]; }
        if (n < u.N && base + lane.y < u.K) { b = q5k_value(u.b_row + n, base + lane.y, u.blocks); }
        tile_a[index] = a;
        tile_b[index] = b;
        workgroupBarrier();
        for (var k = 0u; k < TILE_SIZE; k++) {
            sum = fma(tile_a[lane.y * TILE_SIZE + k], tile_b[k * TILE_SIZE + lane.x], sum);
        }
        workgroupBarrier();
    }
    if (m < u.M && n < u.N) { output[u.c_offset + m * u.N + n] = sum * u.alpha; }
}
