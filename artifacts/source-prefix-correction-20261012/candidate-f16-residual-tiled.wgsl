// Register-Tiled Matrix Multiplication Kernel - f16 weights, f32 activations
//
// High-throughput prefill matmul: f32 activations (A), f16 weights (B), f32 output (C).
// Each thread computes a 4x4 tile of C (16 outputs), vs 1 output in matmul_f16w_f32a.
// Workgroup tile: 64x64 with 16x16 = 256 threads.
// This yields 16x more outputs per workgroup than the base f16w_f32a kernel.
//
// C[M,N] = A[M,K] * B^T[N,K]  (transpose_b=1, RDRR row-major layout)
//
// Design:
// - f32 A storage (activations), f16 B storage (weights)
// - f32 shared tiles for broader Vulkan/Dawn compatibility
// - f32 accumulators for numerical stability
// - B loaded from f16 global into f32 shared tile on staging

enable f16;

// Tile dimensions
const TILE_M: u32 = 64u;   // Rows of C per workgroup
const TILE_N: u32 = 64u;   // Cols of C per workgroup
const TILE_K: u32 = 16u;   // K-dimension tile step

// Thread tile: each thread computes a 4x4 block of C
const THREAD_M: u32 = 4u;
const THREAD_N: u32 = 4u;

// Workgroup dimensions: 16x16 = 256 threads
const WG_M: u32 = 16u;
const WG_N: u32 = 16u;

// Shared memory: TILE_M * TILE_K + TILE_N * TILE_K
// tileA: 64*16 f32 = 4096 bytes, tileB: 64*16 f32 = 4096 bytes → 8192 bytes total
var<workgroup> tileA: array<f32, 1024>;  // TILE_M * TILE_K = 64 * 16
var<workgroup> tileB: array<f32, 1024>;  // TILE_N * TILE_K = 64 * 16

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    transpose_b: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f16>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

fn add_u64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low = a.x + b.x;
    return vec2<u32>(low, a.y + b.y + u32(low < a.x));
}
fn negate_u64(a: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(0u - a.x, ~a.y + u32(a.x == 0u));
}
fn aligned_significand(value: f32, exponent: u32) -> vec2<u32> {
    let bits = bitcast<u32>(value);
    let stored_exponent = (bits >> 23u) & 255u;
    let mantissa = (bits & 0x007fffffu) | select(0u, 0x00800000u, stored_exponent != 0u);
    let shift = exponent - max(stored_exponent, 1u);
    var magnitude = vec2<u32>(0u);
    if (shift == 0u) { magnitude.y = mantissa; }
    else if (shift < 32u) { magnitude = vec2<u32>(mantissa << (32u - shift), mantissa >> shift); }
    else if (shift < 64u) { magnitude.x = mantissa >> (shift - 32u); }
    return select(magnitude, negate_u64(magnitude), (bits & 0x80000000u) != 0u);
}
fn addition_residual(a: f32, b: f32, sum: f32) -> f32 {
    if (sum == a) { return b; }
    if (sum == b) { return a; }
    let exponent = max(max((bitcast<u32>(a) >> 23u) & 255u, (bitcast<u32>(b) >> 23u) & 255u),
                       max((bitcast<u32>(sum) >> 23u) & 255u, 1u));
    if (exponent == 255u) { return 0.0; }
    let residual = add_u64(add_u64(aligned_significand(a, exponent), aligned_significand(b, exponent)),
                            negate_u64(aligned_significand(sum, exponent)));
    let negative = (residual.y & 0x80000000u) != 0u;
    let magnitude = select(residual, negate_u64(residual), negative);
    let value = ldexp(f32(magnitude.x), i32(exponent) - 182);
    return select(value, -value, negative);
}


// Construct the binary32 result, including nearest-even subnormal rounding,
// without an arithmetic operation that may flush subnormal operands/results.
fn rounded_shift(value: vec2<u32>, shift: u32) -> u32 {
    if (shift == 0u) { return value.x; }
    if (shift >= 64u) { return 0u; }
    var quotient: u32;
    var above: bool;
    var tie: bool;
    if (shift < 32u) {
        quotient = (value.x >> shift) | (value.y << (32u - shift));
        let remainder = value.x & ((1u << shift) - 1u);
        let halfway = 1u << (shift - 1u);
        above = remainder > halfway; tie = remainder == halfway;
    } else if (shift == 32u) {
        quotient = value.y; above = value.x > 0x80000000u; tie = value.x == 0x80000000u;
    } else {
        let high_shift = shift - 32u;
        quotient = value.y >> high_shift;
        let remainder = value.y & ((1u << high_shift) - 1u);
        let halfway = 1u << (high_shift - 1u);
        above = remainder > halfway || (remainder == halfway && value.x != 0u);
        tie = remainder == halfway && value.x == 0u;
    }
    return quotient + u32(above || (tie && (quotient & 1u) != 0u));
}
fn scaled_significand(magnitude: vec2<u32>, exponent: i32, negative: bool) -> f32 {
    let sign = select(0u, 0x80000000u, negative);
    if (all(magnitude == vec2<u32>(0u))) { return bitcast<f32>(sign); }
    let top = select(firstLeadingBit(magnitude.x), firstLeadingBit(magnitude.y) + 32u, magnitude.y != 0u);
    var unbiased = exponent + i32(top);
    var bits: u32;
    if (unbiased >= -126) {
        var significand = select(magnitude.x << (23u - min(top, 23u)), rounded_shift(magnitude, top - min(top, 23u)), top > 23u);
        if (significand >= 0x01000000u) { significand >>= 1u; unbiased += 1; }
        bits = (u32(unbiased + 127) << 23u) | (significand & 0x007fffffu);
    } else {
        let shift = exponent + 149;
        if (shift >= 0) { bits = magnitude.x << u32(shift); }
        else { bits = rounded_shift(magnitude, u32(-shift)); }
    }
    return bitcast<f32>(sign | bits);
}

// Exact product residual from the 24-bit significands, avoiding a floating
// multiply/subtract identity that a relaxed backend may reassociate.
fn product_residual(a: f32, b: f32, product: f32) -> f32 {
    let ab = bitcast<u32>(a); let bb = bitcast<u32>(b); let pb = bitcast<u32>(product);
    let ae = (ab >> 23u) & 255u; let be = (bb >> 23u) & 255u; let pe = (pb >> 23u) & 255u;
    if (ae == 255u || be == 255u || pe == 255u || (ab & 0x7fffffffu) == 0u || (bb & 0x7fffffffu) == 0u) { return 0.0; }
    let am = (ab & 0x007fffffu) | select(0u, 0x00800000u, ae != 0u);
    let bm = (bb & 0x007fffffu) | select(0u, 0x00800000u, be != 0u);
    let pm = (pb & 0x007fffffu) | select(0u, 0x00800000u, pe != 0u);
    let al = am & 65535u; let ah = am >> 16u;
    let bl = bm & 65535u; let bh = bm >> 16u;
    let cross = al * bh + ah * bl;
    let exact = add_u64(vec2<u32>(al * bl, ah * bh), vec2<u32>(cross << 16u, cross >> 16u));
    if ((pb & 0x7fffffffu) == 0u) {
        return scaled_significand(exact, i32(max(ae, 1u)) + i32(max(be, 1u)) - 300, ((ab ^ bb) & 0x80000000u) != 0u);
    }
    let shift = i32(max(pe, 1u)) + 150 - i32(max(ae, 1u)) - i32(max(be, 1u));
    if (shift >= 64) { return bitcast<f32>(pb ^ 0x80000000u); }
    if (shift < 0) { return 0.0; }
    var rounded = vec2<u32>(0u);
    if (shift == 0) { rounded.x = pm; }
    else if (shift < 32) { rounded = vec2<u32>(pm << u32(shift), pm >> (32u - u32(shift))); }
    else { rounded.y = pm << (u32(shift) - 32u); }
    let residual = add_u64(exact, negate_u64(rounded));
    let negative = (residual.y & 0x80000000u) != 0u;
    let magnitude = select(residual, negate_u64(residual), negative);
    let sign = negative != (((ab ^ bb) & 0x80000000u) != 0u);
    return scaled_significand(magnitude, i32(max(ae, 1u)) + i32(max(be, 1u)) - 300, sign);
}


@compute @workgroup_size(WG_M, WG_N, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tx = local_id.x;  // 0..15, maps to M dimension
    let ty = local_id.y;  // 0..15, maps to N dimension
    let tid = tx * WG_N + ty;  // Linear thread index 0..255

    // Global row/col base for this thread's 4x4 output tile
    let row_base = wg_id.x * TILE_M + tx * THREAD_M;
    let col_base = wg_id.y * TILE_N + ty * THREAD_N;

    // f32 accumulators: 4x4 = 16 per thread
    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    var error00: f32 = 0.0;
    var error01: f32 = 0.0;
    var error02: f32 = 0.0;
    var error03: f32 = 0.0;
    var error10: f32 = 0.0;
    var error11: f32 = 0.0;
    var error12: f32 = 0.0;
    var error13: f32 = 0.0;
    var error20: f32 = 0.0;
    var error21: f32 = 0.0;
    var error22: f32 = 0.0;
    var error23: f32 = 0.0;
    var error30: f32 = 0.0;
    var error31: f32 = 0.0;
    var error32: f32 = 0.0;
    var error33: f32 = 0.0;

    let num_tiles = (u.K + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let k_offset = t * TILE_K;

        // Load A tile (f32): 64 rows x 16 cols = 1024 elements, 256 threads → 4 each
        {
            let load_base = tid * 4u;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let elem_idx = load_base + i;
                let load_row = elem_idx / TILE_K;
                let load_col = elem_idx % TILE_K;
                let global_row = wg_id.x * TILE_M + load_row;
                let global_col = k_offset + load_col;
                if (global_row < u.M && global_col < u.K) {
                    tileA[elem_idx] = A[global_row * u.K + global_col];
                } else {
                    tileA[elem_idx] = 0.0;
                }
            }
        }

        // Load B tile (f16 → f32): B is [N, K] (row-major/transposed)
        {
            let load_base = tid * 4u;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let elem_idx = load_base + i;
                let load_row = elem_idx / TILE_K;  // N-dimension local offset
                let load_col = elem_idx % TILE_K;  // K-dimension local offset
                let global_n = wg_id.y * TILE_N + load_row;
                let global_k = k_offset + load_col;
                if (global_n < u.N && global_k < u.K) {
                    tileB[elem_idx] = f32(B[global_n * u.K + global_k]);
                } else {
                    tileB[elem_idx] = 0.0;
                }
            }
        }

        workgroupBarrier();

        // Compute: each thread accumulates its 4x4 output tile
        for (var k: u32 = 0u; k < TILE_K; k = k + 1u) {
            let a0 = tileA[(tx * THREAD_M + 0u) * TILE_K + k];
            let a1 = tileA[(tx * THREAD_M + 1u) * TILE_K + k];
            let a2 = tileA[(tx * THREAD_M + 2u) * TILE_K + k];
            let a3 = tileA[(tx * THREAD_M + 3u) * TILE_K + k];

            let b0 = tileB[(ty * THREAD_N + 0u) * TILE_K + k];
            let b1 = tileB[(ty * THREAD_N + 1u) * TILE_K + k];
            let b2 = tileB[(ty * THREAD_N + 2u) * TILE_K + k];
            let b3 = tileB[(ty * THREAD_N + 3u) * TILE_K + k];

            {
                let product = a0 * b0;
                let total = acc00 + product;
                error00 += product_residual(a0, b0, product) + addition_residual(acc00, product, total);
                acc00 = total;
            } {
                let product = a0 * b1;
                let total = acc01 + product;
                error01 += product_residual(a0, b1, product) + addition_residual(acc01, product, total);
                acc01 = total;
            } {
                let product = a0 * b2;
                let total = acc02 + product;
                error02 += product_residual(a0, b2, product) + addition_residual(acc02, product, total);
                acc02 = total;
            } {
                let product = a0 * b3;
                let total = acc03 + product;
                error03 += product_residual(a0, b3, product) + addition_residual(acc03, product, total);
                acc03 = total;
            }
            {
                let product = a1 * b0;
                let total = acc10 + product;
                error10 += product_residual(a1, b0, product) + addition_residual(acc10, product, total);
                acc10 = total;
            } {
                let product = a1 * b1;
                let total = acc11 + product;
                error11 += product_residual(a1, b1, product) + addition_residual(acc11, product, total);
                acc11 = total;
            } {
                let product = a1 * b2;
                let total = acc12 + product;
                error12 += product_residual(a1, b2, product) + addition_residual(acc12, product, total);
                acc12 = total;
            } {
                let product = a1 * b3;
                let total = acc13 + product;
                error13 += product_residual(a1, b3, product) + addition_residual(acc13, product, total);
                acc13 = total;
            }
            {
                let product = a2 * b0;
                let total = acc20 + product;
                error20 += product_residual(a2, b0, product) + addition_residual(acc20, product, total);
                acc20 = total;
            } {
                let product = a2 * b1;
                let total = acc21 + product;
                error21 += product_residual(a2, b1, product) + addition_residual(acc21, product, total);
                acc21 = total;
            } {
                let product = a2 * b2;
                let total = acc22 + product;
                error22 += product_residual(a2, b2, product) + addition_residual(acc22, product, total);
                acc22 = total;
            } {
                let product = a2 * b3;
                let total = acc23 + product;
                error23 += product_residual(a2, b3, product) + addition_residual(acc23, product, total);
                acc23 = total;
            }
            {
                let product = a3 * b0;
                let total = acc30 + product;
                error30 += product_residual(a3, b0, product) + addition_residual(acc30, product, total);
                acc30 = total;
            } {
                let product = a3 * b1;
                let total = acc31 + product;
                error31 += product_residual(a3, b1, product) + addition_residual(acc31, product, total);
                acc31 = total;
            } {
                let product = a3 * b2;
                let total = acc32 + product;
                error32 += product_residual(a3, b2, product) + addition_residual(acc32, product, total);
                acc32 = total;
            } {
                let product = a3 * b3;
                let total = acc33 + product;
                error33 += product_residual(a3, b3, product) + addition_residual(acc33, product, total);
                acc33 = total;
            }
        }

           
           
           
           

        workgroupBarrier();
    }

    // Write 4x4 output tile (f32)
    let alpha = u.alpha;
    if (row_base + 0u < u.M && col_base + 0u < u.N) { C[(row_base + 0u) * u.N + col_base + 0u] = (acc00 + error00) * alpha; }
    if (row_base + 0u < u.M && col_base + 1u < u.N) { C[(row_base + 0u) * u.N + col_base + 1u] = (acc01 + error01) * alpha; }
    if (row_base + 0u < u.M && col_base + 2u < u.N) { C[(row_base + 0u) * u.N + col_base + 2u] = (acc02 + error02) * alpha; }
    if (row_base + 0u < u.M && col_base + 3u < u.N) { C[(row_base + 0u) * u.N + col_base + 3u] = (acc03 + error03) * alpha; }

    if (row_base + 1u < u.M && col_base + 0u < u.N) { C[(row_base + 1u) * u.N + col_base + 0u] = (acc10 + error10) * alpha; }
    if (row_base + 1u < u.M && col_base + 1u < u.N) { C[(row_base + 1u) * u.N + col_base + 1u] = (acc11 + error11) * alpha; }
    if (row_base + 1u < u.M && col_base + 2u < u.N) { C[(row_base + 1u) * u.N + col_base + 2u] = (acc12 + error12) * alpha; }
    if (row_base + 1u < u.M && col_base + 3u < u.N) { C[(row_base + 1u) * u.N + col_base + 3u] = (acc13 + error13) * alpha; }

    if (row_base + 2u < u.M && col_base + 0u < u.N) { C[(row_base + 2u) * u.N + col_base + 0u] = (acc20 + error20) * alpha; }
    if (row_base + 2u < u.M && col_base + 1u < u.N) { C[(row_base + 2u) * u.N + col_base + 1u] = (acc21 + error21) * alpha; }
    if (row_base + 2u < u.M && col_base + 2u < u.N) { C[(row_base + 2u) * u.N + col_base + 2u] = (acc22 + error22) * alpha; }
    if (row_base + 2u < u.M && col_base + 3u < u.N) { C[(row_base + 2u) * u.N + col_base + 3u] = (acc23 + error23) * alpha; }

    if (row_base + 3u < u.M && col_base + 0u < u.N) { C[(row_base + 3u) * u.N + col_base + 0u] = (acc30 + error30) * alpha; }
    if (row_base + 3u < u.M && col_base + 1u < u.N) { C[(row_base + 3u) * u.N + col_base + 1u] = (acc31 + error31) * alpha; }
    if (row_base + 3u < u.M && col_base + 2u < u.N) { C[(row_base + 3u) * u.N + col_base + 2u] = (acc32 + error32) * alpha; }
    if (row_base + 3u < u.M && col_base + 3u < u.N) { C[(row_base + 3u) * u.N + col_base + 3u] = (acc33 + error33) * alpha; }
}
