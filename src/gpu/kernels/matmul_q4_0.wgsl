// AUTO-GENERATED from src/gpu/kernels/codegen/q4_0_common.wgslinc.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// Cooperative quarter-block GEMV adapted from Pooled engine/wgsl/coop.js (6b4a24f7).
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
            sum = fma(input[u.a_offset + m * u.K + k], q40_value(u.b_row + n, k, u.blocks), sum);
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
        if (n < u.N && base + lane.y < u.K) { b = q40_value(u.b_row + n, base + lane.y, u.blocks); }
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

// Eight original values per invocation. Q6_K's segment is a logical 32-value
// slice within its 256-value block; other formats have one such slice.
fn packed_word(offset: u32) -> u32 {
    let shift = (offset % 4u) * 8u;
    let low = weights[offset / 4u];
    if (shift == 0u) { return low; }
    return (low >> shift) | (weights[offset / 4u + 1u] << (32u - shift));
}
fn packed_bytes(word: u32) -> vec4<u32> {
    return (vec4<u32>(word) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(255u);
}
fn signed_bytes(word: u32) -> vec4<f32> {
    let w = bitcast<i32>(word);
    return vec4<f32>(f32((w << 24u) >> 24u), f32((w << 16u) >> 24u),
        f32((w << 8u) >> 24u), f32(w >> 24u));
}
fn packed_quarter_term(base: u32, format: u32, segment: u32, quarter: u32,
                       xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    if (format == 3u) {
        let section = segment % 4u;
        let low_offset = base + (segment / 4u) * 64u + (section % 2u) * 32u + quarter * 4u;
        let high_offset = base + 128u + (segment / 4u) * 32u + quarter * 4u;
        let low_shift = vec4<u32>((section / 2u) * 4u);
        let high_shift = vec4<u32>(section * 2u);
        let qa = ((packed_bytes(packed_word(low_offset)) >> low_shift) & vec4<u32>(15u))
            | (((packed_bytes(packed_word(high_offset)) >> high_shift) & vec4<u32>(3u)) << vec4<u32>(4u));
        let qb = ((packed_bytes(packed_word(low_offset + 16u)) >> low_shift) & vec4<u32>(15u))
            | (((packed_bytes(packed_word(high_offset + 16u)) >> high_shift) & vec4<u32>(3u)) << vec4<u32>(4u));
        let d = packed_half(base + 208u);
        let da = d * f32(packed_signed_byte(base + 192u + segment * 2u));
        let db = d * f32(packed_signed_byte(base + 193u + segment * 2u));
        return dot(da * (vec4<f32>(qa) - vec4<f32>(32.0)), xa)
            + dot(db * (vec4<f32>(qb) - vec4<f32>(32.0)), xb);
    }
    let scale = packed_half(base);
    if (format == 2u) {
        let first = base + 2u + quarter * 8u;
        return scale * (dot(signed_bytes(packed_word(first)), xa)
            + dot(signed_bytes(packed_word(first + 4u)), xb));
    }
    var data_offset = 2u;
    if (format == 1u) { data_offset = 4u; }
    if (format == 4u) { data_offset = 6u; }
    let bytes = packed_bytes(packed_word(base + data_offset + quarter * 4u));
    var lo = bytes & vec4<u32>(15u);
    var hi = bytes >> vec4<u32>(4u);
    if (format == 4u) {
        let high_bits = vec4<u32>(packed_word(base + 2u));
        let bit = vec4<u32>(quarter * 4u) + vec4<u32>(0u, 1u, 2u, 3u);
        lo |= ((high_bits >> bit) & vec4<u32>(1u)) << vec4<u32>(4u);
        hi |= ((high_bits >> (bit + vec4<u32>(16u))) & vec4<u32>(1u)) << vec4<u32>(4u);
        return scale * (dot(vec4<f32>(lo) - vec4<f32>(16.0), xa)
            + dot(vec4<f32>(hi) - vec4<f32>(16.0), xb));
    }
    if (format == 1u) {
        let minimum = vec4<f32>(packed_half(base + 2u));
        // Keep the additive minimum per logical value, including negative minima.
        return dot(scale * vec4<f32>(lo) + minimum, xa)
            + dot(scale * vec4<f32>(hi) + minimum, xb);
    }
    return scale * (dot(vec4<f32>(lo) - vec4<f32>(8.0), xa)
        + dot(vec4<f32>(hi) - vec4<f32>(8.0), xb));
}

fn cooperative_term(row: u32, slice: u32, quarter: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    if (row >= u.b_row + u.N) { return 0.0; }
    let slices_per_block = BLOCK_VALUES / 32u;
    let base = (row * u.blocks + slice / slices_per_block) * BLOCK_BYTES;
    return packed_quarter_term(base, QUANT_FORMAT, slice % slices_per_block, quarter, xa, xb);
}
var<workgroup> cooperative_partial: array<vec4<f32>, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn cooperative(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>,
               @builtin(local_invocation_index) lane: u32) {
    let tiles_n = (u.N + COLS_PER_WG - 1u) / COLS_PER_WG;
    let tile = group.x + group.y * grid.x;
    let m = tile / tiles_n;
    if (m >= u.M) { return; }
    let row = u.b_row + (tile % tiles_n) * COLS_PER_WG;
    let quarter = lane % 4u;
    var sums = vec4<f32>(0.0);
    for (var b = lane / 4u; b < u.K / 32u; b += WORKGROUP_SIZE / 4u) {
        let first = select(quarter * 4u, quarter * 8u, QUANT_FORMAT == 2u);
        let second = select(first + 16u, first + 4u, QUANT_FORMAT == 2u);
        let x = u.a_offset + m * u.K + b * 32u;
        let xa = vec4<f32>(input[x+first], input[x+first+1u], input[x+first+2u], input[x+first+3u]);
        let xb = vec4<f32>(input[x+second], input[x+second+1u], input[x+second+2u], input[x+second+3u]);
        sums += vec4<f32>(cooperative_term(row, b, quarter, xa, xb),
            cooperative_term(row+1u, b, quarter, xa, xb), cooperative_term(row+2u, b, quarter, xa, xb),
            cooperative_term(row+3u, b, quarter, xa, xb));
    }
    cooperative_partial[lane] = sums;
    workgroupBarrier();
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (lane < stride) { cooperative_partial[lane] += cooperative_partial[lane + stride]; }
        workgroupBarrier();
    }
    if (lane < COLS_PER_WG && row + lane < u.b_row + u.N) {
        output[u.c_offset + m * u.N + row - u.b_row + lane] = cooperative_partial[0][lane] * u.alpha;
    }
}
