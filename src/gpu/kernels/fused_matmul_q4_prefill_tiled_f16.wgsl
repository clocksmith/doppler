// Register-Tiled Q4_K Matmul for Prefill — F32 activations, F16 output, Q4_K weights
//
// Target shape: M >= 16 (prefill), Q4_K weights, F32 activations, F16 output.
// Matches q4_fused_batched_f16 binding contract (A=f32, C=f16) so it slots
// into the existing Q4_K matmul selection path when the caller has already
// resolved to that variant.
//
// C[M,N] = A[M,K] * dequant(B_q4k)^T  (transpose_b implicit: weights row-major [N, K])
//
// Design:
// - 16 x 16 = 256 threads per workgroup
// - Each thread computes a 4 x 4 register tile of outputs (16 outputs)
// - Per workgroup: TILE_M=64 rows x TILE_N=64 cols = 4096 outputs
// - K-tile: 16 elements per iteration (TILE_K divides QK_K=256)
// - Shared memory: tileA (64*16 f32) + tileB (64*16 f16) = 6 KB
// - Q4_K dequantization runs inline when loading tileB, reusing helpers from
//   fused_matmul_q4_batched_f16.wgsl. Weights are read once into shared
//   memory and then reused across the 4 x 4 register tile, amortizing
//   dequant cost 16x vs per-output dequant kernels.
//
// Dispatch convention (Q4_K): workgroupsX = ceil(N / colsPerWg=64),
// workgroupsY = ceil(M / tileM=64). wg_id.x indexes N-column tile;
// wg_id.y indexes M-row tile.
//
// For Gemma 4 E2B prefill (M=64, N=6144, K=1536):
//   workgroups = (96, 1, 1) = 96 WGs per matmul
// vs fused_matmul_q4_batched_f16 (TILE_M=4, 1 col/WG): 98,304 WGs — 1024x reduction.

enable f16;

const QK_K: u32 = 256u;
const SUBBLOCK_SIZE: u32 = 32u;

const TILE_M: u32 = 64u;
const TILE_N: u32 = 64u;
const TILE_K: u32 = 16u;

const THREAD_M: u32 = 4u;
const THREAD_N: u32 = 4u;

const WG_M: u32 = 16u;  // 16 threads along M covers 64 rows (4 per thread)
const WG_N: u32 = 16u;  // 16 threads along N covers 64 cols (4 per thread)

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
@group(0) @binding(4) var<storage, read_write> C: array<f16>;

var<workgroup> tileA: array<f32, 1024>;  // TILE_M * TILE_K = 64 * 16
var<workgroup> tileB: array<f16, 1024>;  // TILE_N * TILE_K = 64 * 16

fn unpack_f16_lo(packed: u32) -> f32 {
    return unpack2x16float(packed).x;
}

fn unpack_f16_hi(packed: u32) -> f32 {
    return unpack2x16float(packed).y;
}

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

fn get_q4(qs: array<u32, 32>, idx: u32) -> u32 {
    let chunk = idx / 64u;
    let pos_in_chunk = idx % 64u;
    let use_upper = pos_in_chunk >= 32u;
    let byte_in_range = select(pos_in_chunk, pos_in_chunk - 32u, use_upper);
    let byte_idx = chunk * 32u + byte_in_range;

    let word_idx = byte_idx / 4u;
    let byte_in_word = byte_idx % 4u;
    let byte_val = (qs[word_idx] >> (byte_in_word * 8u)) & 0xFFu;

    if (use_upper) {
        return (byte_val >> 4u) & 0xFu;
    } else {
        return byte_val & 0xFu;
    }
}

@compute @workgroup_size(WG_M, WG_N, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let tx = lid.x;  // 0..15, row-local
    let ty = lid.y;  // 0..15, col-local
    let tid = tx * WG_N + ty;  // 0..255, linear thread id

    // Q4K dispatch convention: workgroupsX = ceil(N/colsPerWg), workgroupsY = ceil(M/tileM).
    // wg_id.x indexes the N-column tile; wg_id.y indexes the M-row tile.
    let row_base = wg_id.y * TILE_M + tx * THREAD_M;
    let col_base = wg_id.x * TILE_N + ty * THREAD_N;

    // 4x4 f32 accumulators (16 per thread)
    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    let num_k_tiles = (u.K + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < num_k_tiles; t = t + 1u) {
        let k_offset = t * TILE_K;

        // ===== Stage 1: Load A tile (f16) =====
        // 64 rows x 16 K = 1024 elements across 256 threads = 4 per thread.
        {
            let base = tid * 4u;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let idx = base + i;
                let load_row = idx / TILE_K;
                let load_k = idx % TILE_K;
                let global_row = wg_id.y * TILE_M + load_row;
                let global_k = k_offset + load_k;
                if (global_row < u.M && global_k < u.K) {
                    tileA[idx] = A[global_row * u.K + global_k];
                } else {
                    tileA[idx] = 0.0;
                }
            }
        }

        // ===== Stage 2: Load B tile with inline Q4_K dequant =====
        // 64 cols x 16 K = 1024 slots across 256 threads = 4 per thread.
        // Layout: tileB[col_local * TILE_K + k_local]. Thread tid's 4 slots
        // all belong to col_local = tid / 4, with k_local spanning
        // [(tid % 4) * 4, (tid % 4) * 4 + 3]. k_offset is a multiple of
        // TILE_K=16 and TILE_K divides QK_K=256, so all 4 slots fall within
        // the same Q4_K block (no block-crossing per thread).
        {
            let base = tid * 4u;
            let col_local = base / TILE_K;
            let k_local_start = base % TILE_K;
            let global_col = wg_id.x * TILE_N + col_local;

            if (global_col < u.N) {
                let global_k_base = k_offset + k_local_start;
                let block_idx = global_k_base / QK_K;
                let elem_base = global_k_base - block_idx * QK_K;

                let block = B_q4k[global_col * u.num_blocks_per_row + block_idx];
                let d = unpack_f16_lo(block.d_dmin);
                let dmin = unpack_f16_hi(block.d_dmin);

                var last_sb: u32 = 0xFFFFFFFFu;
                var scale: f32 = 0.0;
                var min_val: f32 = 0.0;

                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    let elem = elem_base + i;
                    let sb = elem / SUBBLOCK_SIZE;
                    if (sb != last_sb) {
                        let sm = get_scale_min_k4(block.scales, sb);
                        scale = d * f32(sm.x);
                        min_val = dmin * f32(sm.y);
                        last_sb = sb;
                    }
                    let q = get_q4(block.qs, elem);
                    let w = scale * f32(q) - min_val;
                    let slot = col_local * TILE_K + k_local_start + i;
                    let global_k = k_offset + k_local_start + i;
                    if (global_k < u.K) {
                        tileB[slot] = f16(w);
                    } else {
                        tileB[slot] = f16(0.0);
                    }
                }
            } else {
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    let slot = col_local * TILE_K + k_local_start + i;
                    tileB[slot] = f16(0.0);
                }
            }
        }

        workgroupBarrier();

        // ===== Stage 3: Register-tile matmul =====
        // tileA: [TILE_M=64, TILE_K=16], indexed as tileA[row_local * TILE_K + k].
        // tileB: [TILE_N=64, TILE_K=16], indexed as tileB[col_local * TILE_K + k].
        // Each thread reads 4 A rows and 4 B cols per k, 4x4 outer product.
        for (var k: u32 = 0u; k < TILE_K; k = k + 1u) {
            let a0 = tileA[(tx * THREAD_M + 0u) * TILE_K + k];
            let a1 = tileA[(tx * THREAD_M + 1u) * TILE_K + k];
            let a2 = tileA[(tx * THREAD_M + 2u) * TILE_K + k];
            let a3 = tileA[(tx * THREAD_M + 3u) * TILE_K + k];

            let b0 = f32(tileB[(ty * THREAD_N + 0u) * TILE_K + k]);
            let b1 = f32(tileB[(ty * THREAD_N + 1u) * TILE_K + k]);
            let b2 = f32(tileB[(ty * THREAD_N + 2u) * TILE_K + k]);
            let b3 = f32(tileB[(ty * THREAD_N + 3u) * TILE_K + k]);

            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }

        workgroupBarrier();
    }

    // ===== Stage 4: Write 4x4 output tile =====
    let alpha = u.alpha;
    if (row_base + 0u < u.M && col_base + 0u < u.N) { C[(row_base + 0u) * u.N + col_base + 0u] = f16(acc00 * alpha); }
    if (row_base + 0u < u.M && col_base + 1u < u.N) { C[(row_base + 0u) * u.N + col_base + 1u] = f16(acc01 * alpha); }
    if (row_base + 0u < u.M && col_base + 2u < u.N) { C[(row_base + 0u) * u.N + col_base + 2u] = f16(acc02 * alpha); }
    if (row_base + 0u < u.M && col_base + 3u < u.N) { C[(row_base + 0u) * u.N + col_base + 3u] = f16(acc03 * alpha); }

    if (row_base + 1u < u.M && col_base + 0u < u.N) { C[(row_base + 1u) * u.N + col_base + 0u] = f16(acc10 * alpha); }
    if (row_base + 1u < u.M && col_base + 1u < u.N) { C[(row_base + 1u) * u.N + col_base + 1u] = f16(acc11 * alpha); }
    if (row_base + 1u < u.M && col_base + 2u < u.N) { C[(row_base + 1u) * u.N + col_base + 2u] = f16(acc12 * alpha); }
    if (row_base + 1u < u.M && col_base + 3u < u.N) { C[(row_base + 1u) * u.N + col_base + 3u] = f16(acc13 * alpha); }

    if (row_base + 2u < u.M && col_base + 0u < u.N) { C[(row_base + 2u) * u.N + col_base + 0u] = f16(acc20 * alpha); }
    if (row_base + 2u < u.M && col_base + 1u < u.N) { C[(row_base + 2u) * u.N + col_base + 1u] = f16(acc21 * alpha); }
    if (row_base + 2u < u.M && col_base + 2u < u.N) { C[(row_base + 2u) * u.N + col_base + 2u] = f16(acc22 * alpha); }
    if (row_base + 2u < u.M && col_base + 3u < u.N) { C[(row_base + 2u) * u.N + col_base + 3u] = f16(acc23 * alpha); }

    if (row_base + 3u < u.M && col_base + 0u < u.N) { C[(row_base + 3u) * u.N + col_base + 0u] = f16(acc30 * alpha); }
    if (row_base + 3u < u.M && col_base + 1u < u.N) { C[(row_base + 3u) * u.N + col_base + 1u] = f16(acc31 * alpha); }
    if (row_base + 3u < u.M && col_base + 2u < u.N) { C[(row_base + 3u) * u.N + col_base + 2u] = f16(acc32 * alpha); }
    if (row_base + 3u < u.M && col_base + 3u < u.N) { C[(row_base + 3u) * u.N + col_base + 3u] = f16(acc33 * alpha); }
}
