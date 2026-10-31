// AUTO-GENERATED from src/gpu/kernels/codegen/q4_0_common.wgslinc.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
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
