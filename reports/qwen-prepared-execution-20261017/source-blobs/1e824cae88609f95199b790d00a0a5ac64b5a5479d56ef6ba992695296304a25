// AUTO-GENERATED from src/gpu/kernels/fused_matmul_q4_widetile.wgsl.
// Edit the source kernel and src/gpu/kernels/codegen/wgsl-variants.js, then run `npm run kernels:codegen:sync`.
// Fused Q4_K Matmul Kernel — WideTile register pattern (F32 A, Q4_K B, F32 C)
//
// F32-output twin of fused_matmul_q4_widetile_f16.wgsl. Same register-tiling
// pattern and unpack4xU8-vectorized dequant — only the output dtype and the
// final cast differ. Kept as a separate file because the WGSL binding type
// is part of the kernel ABI; callers that want f32 output select this
// variant rather than this kernel coercing behind their back.
//
// See fused_matmul_q4_widetile_f16.wgsl for the full design rationale.

enable f16;

const QK_K: u32 = 256u;
const SUBBLOCK_SIZE: u32 = 32u;
const NUM_SUBBLOCK_PAIRS: u32 = 4u;
const U32S_PER_PAIR: u32 = 8u;

override TILE_M: u32 = 4u;
override TILE_N: u32 = 256u;

const MAX_TILE_M: u32 = 8u;

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    num_blocks_per_row: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Q4KBlock {
    d_dmin: u32,
    scales: array<u32, 3>,
    qs: array<u32, 32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B_q4k: array<Q4KBlock>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

var<workgroup> tileA: array<f32, MAX_TILE_M * QK_K>;

fn unpack_f16_lo(packed: u32) -> f32 { return unpack2x16float(packed).x; }
fn unpack_f16_hi(packed: u32) -> f32 { return unpack2x16float(packed).y; }

fn get_scale_byte(scales: array<u32, 3>, byte_idx: u32) -> u32 {
    let word_idx = byte_idx / 4u;
    let byte_in_word = byte_idx % 4u;
    return (scales[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn get_scale_min_k4(scales: array<u32, 3>, j: u32) -> vec2<u32> {
    var sc: u32;
    var mn: u32;
    if (j < 4u) {
        sc = get_scale_byte(scales, j) & 63u;
        mn = get_scale_byte(scales, j + 4u) & 63u;
    } else {
        let q_j = get_scale_byte(scales, j + 4u);
        let q_lo = get_scale_byte(scales, j - 4u);
        let q_hi = get_scale_byte(scales, j);
        sc = (q_j & 0xFu) | ((q_lo >> 6u) << 4u);
        mn = (q_j >> 4u) | ((q_hi >> 6u) << 4u);
    }
    return vec2<u32>(sc, mn);
}

fn load_a_vec4(row_local: u32, base_elem: u32) -> vec4<f32> {
    let a_row_base = row_local * QK_K;
    return vec4<f32>(
        tileA[a_row_base + base_elem + 0u],
        tileA[a_row_base + base_elem + 1u],
        tileA[a_row_base + base_elem + 2u],
        tileA[a_row_base + base_elem + 3u]
    );
}

@compute @workgroup_size(TILE_N, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    if (TILE_M > MAX_TILE_M) { return; }

    let col_local = lid.x;
    let row_base = wg_id.y * TILE_M;
    let col_base = wg_id.x * TILE_N;
    let col = col_base + col_local;

    // Reduce each quantization block separately to bound F32 dependency depth.
    // Four local FMA lanes plus one outer scalar per row preserve bounded registers.
    var totals: array<f32, MAX_TILE_M>;
    for (var i: u32 = 0u; i < TILE_M; i = i + 1u) {
        totals[i] = 0.0;
    }

    let num_blocks = u.num_blocks_per_row;

    for (var b: u32 = 0u; b < num_blocks; b = b + 1u) {
        var results: array<vec4<f32>, MAX_TILE_M>;
        for (var m = 0u; m < TILE_M; m++) { results[m] = vec4<f32>(0.0); }
        let total_a_elems = TILE_M * QK_K;
        for (var idx: u32 = col_local; idx < total_a_elems; idx = idx + TILE_N) {
            let load_row_local = idx / QK_K;
            let load_k = idx % QK_K;
            let global_row = row_base + load_row_local;
            let global_k = b * QK_K + load_k;
            if (global_row < u.M && global_k < u.K) {
                tileA[idx] = A[global_row * u.K + global_k];
            } else {
                tileA[idx] = 0.0;
            }
        }
        workgroupBarrier();

        if (col < u.N) {
            let block_index = col * num_blocks + b;
            let block_scales = B_q4k[block_index].scales;
            let d_dmin = B_q4k[block_index].d_dmin;
            let d = unpack_f16_lo(d_dmin);
            let dmin = unpack_f16_hi(d_dmin);

            for (var p: u32 = 0u; p < NUM_SUBBLOCK_PAIRS; p = p + 1u) {
                let sm_even = get_scale_min_k4(block_scales, 2u * p);
                let sm_odd = get_scale_min_k4(block_scales, 2u * p + 1u);
                let scale_even = d * f32(sm_even.x);
                let min_even_scalar = dmin * f32(sm_even.y);
                let scale_odd = d * f32(sm_odd.x);
                let min_odd_scalar = dmin * f32(sm_odd.y);
                let min_even = vec4<f32>(min_even_scalar);
                let min_odd = vec4<f32>(min_odd_scalar);

                let pair_base_even = p * 64u;
                let pair_base_odd = p * 64u + SUBBLOCK_SIZE;
                let qs_base = p * U32S_PER_PAIR;

                for (var uidx: u32 = 0u; uidx < U32S_PER_PAIR; uidx = uidx + 1u) {
                    let packed = B_q4k[block_index].qs[qs_base + uidx];
                    let lower_u32 = unpack4xU8(packed & 0x0F0F0F0Fu);
                    let upper_u32 = unpack4xU8((packed >> 4u) & 0x0F0F0F0Fu);
                    let w_even = scale_even * vec4<f32>(lower_u32) - min_even;
                    let w_odd = scale_odd * vec4<f32>(upper_u32) - min_odd;

                    let elem_even = pair_base_even + uidx * 4u;
                    let elem_odd = pair_base_odd + uidx * 4u;

                    if (TILE_M > 0u) {
                        results[0] = fma(load_a_vec4(0u, elem_even), w_even, results[0]);
                        results[0] = fma(load_a_vec4(0u, elem_odd), w_odd, results[0]);
                    }
                    if (TILE_M > 1u) {
                        results[1] = fma(load_a_vec4(1u, elem_even), w_even, results[1]);
                        results[1] = fma(load_a_vec4(1u, elem_odd), w_odd, results[1]);
                    }
                    if (TILE_M > 2u) {
                        results[2] = fma(load_a_vec4(2u, elem_even), w_even, results[2]);
                        results[2] = fma(load_a_vec4(2u, elem_odd), w_odd, results[2]);
                    }
                    if (TILE_M > 3u) {
                        results[3] = fma(load_a_vec4(3u, elem_even), w_even, results[3]);
                        results[3] = fma(load_a_vec4(3u, elem_odd), w_odd, results[3]);
                    }
                    if (TILE_M > 4u) {
                        results[4] = fma(load_a_vec4(4u, elem_even), w_even, results[4]);
                        results[4] = fma(load_a_vec4(4u, elem_odd), w_odd, results[4]);
                    }
                    if (TILE_M > 5u) {
                        results[5] = fma(load_a_vec4(5u, elem_even), w_even, results[5]);
                        results[5] = fma(load_a_vec4(5u, elem_odd), w_odd, results[5]);
                    }
                    if (TILE_M > 6u) {
                        results[6] = fma(load_a_vec4(6u, elem_even), w_even, results[6]);
                        results[6] = fma(load_a_vec4(6u, elem_odd), w_odd, results[6]);
                    }
                    if (TILE_M > 7u) {
                        results[7] = fma(load_a_vec4(7u, elem_even), w_even, results[7]);
                        results[7] = fma(load_a_vec4(7u, elem_odd), w_odd, results[7]);
                    }
                }
            }
            for (var m = 0u; m < TILE_M; m++) {
                var subtotal = 0.0;
                for (var lane = 0u; lane < 4u; lane++) { subtotal += results[m][lane]; }
                totals[m] += subtotal;
            }
        }
        workgroupBarrier();
    }

    if (col < u.N) {
        for (var m: u32 = 0u; m < TILE_M; m = m + 1u) {
            let row = row_base + m;
            if (row < u.M) {
                C[row * u.N + col] = totals[m] * u.alpha;
            }
        }
    }
}
