// Register-Tiled Matrix Multiplication Kernel - FP16
//
// High-throughput matmul for prefill using 4x4 register tiling.
// Each thread computes a 4x4 tile of C (16 outputs), vs 1 output in matmul_f16.
// Workgroup tile: 64x64 with 16x16 = 256 threads.
// This yields 16x more outputs per workgroup than the base f16 kernel.
//
// C[M,N] = A[M,K] * B^T[N,K]  (transpose_b=1, SafeTensors/RDRR row-major layout)
//
// Design:
// - f16 storage, f32 accumulators for numerical stability
// - coalesced scalar loads into shared tiles
// - Shared memory tiles: 64 x TILE_K for A and B
// - B is always assumed row-major (transpose_b=1) for RDRR models

enable f16;

// Tile dimensions
const TILE_M: u32 = 64u;   // Rows of C per workgroup
const TILE_N: u32 = 64u;   // Cols of C per workgroup
const TILE_K: u32 = 16u;   // K-dimension tile step

// Thread tile: each thread computes a 4x4 block of C
const THREAD_M: u32 = 4u;
const THREAD_N: u32 = 4u;

// Workgroup dimensions: 16x16 = 256 threads
// 16 threads along M covers 64 rows (4 per thread)
// 16 threads along N covers 64 cols (4 per thread)
const WG_M: u32 = 16u;
const WG_N: u32 = 16u;

// Shared memory: TILE_M * TILE_K + TILE_N * TILE_K
// = 64*16 + 64*16 = 2048 f16 elements = 4096 bytes total
var<workgroup> tileA: array<f16, 1024>;  // TILE_M * TILE_K = 64 * 16
var<workgroup> tileB: array<f16, 1024>;  // TILE_N * TILE_K = 64 * 16

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    transpose_b: u32, // reserved for uniform compatibility; tiled path assumes transpose_b=1
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f16>;
@group(0) @binding(2) var<storage, read> B: array<f16>;
@group(0) @binding(3) var<storage, read_write> C: array<f16>;

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

    let num_tiles = (u.K + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let k_offset = t * TILE_K;

        // Load A tile: 64 rows x 16 cols = 1024 elements, 256 threads -> 4 elements each
        // Thread tid loads elements at indices: tid*4, tid*4+1, tid*4+2, tid*4+3
        // Mapping: element i -> row = i / TILE_K, col = i % TILE_K
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
                    tileA[elem_idx] = f16(0.0);
                }
            }
        }

        // Load B tile: 64 rows x 16 cols = 1024 elements
        // B is [N, K] (row-major/transposed): B[col, k] = B[col * K + k]
        // tileB layout: tileB[n_local * TILE_K + k_local] = B[global_n, global_k]
        {
            let load_base = tid * 4u;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let elem_idx = load_base + i;
                let load_row = elem_idx / TILE_K;  // N-dimension local offset
                let load_col = elem_idx % TILE_K;  // K-dimension local offset
                let global_n = wg_id.y * TILE_N + load_row;
                let global_k = k_offset + load_col;
                if (global_n < u.N && global_k < u.K) {
                    tileB[elem_idx] = B[global_n * u.K + global_k];
                } else {
                    tileB[elem_idx] = f16(0.0);
                }
            }
        }

        workgroupBarrier();

        // Compute: each thread accumulates its 4x4 output tile
        // A tile: tileA[m_local * TILE_K + k] for rows tx*4..tx*4+3
        // B tile: tileB[n_local * TILE_K + k] for cols ty*4..ty*4+3
        for (var k: u32 = 0u; k < TILE_K; k = k + 1u) {
            // Load 4 A values for this thread's rows
            let a0 = f32(tileA[(tx * THREAD_M + 0u) * TILE_K + k]);
            let a1 = f32(tileA[(tx * THREAD_M + 1u) * TILE_K + k]);
            let a2 = f32(tileA[(tx * THREAD_M + 2u) * TILE_K + k]);
            let a3 = f32(tileA[(tx * THREAD_M + 3u) * TILE_K + k]);

            // Load 4 B values for this thread's columns
            let b0 = f32(tileB[(ty * THREAD_N + 0u) * TILE_K + k]);
            let b1 = f32(tileB[(ty * THREAD_N + 1u) * TILE_K + k]);
            let b2 = f32(tileB[(ty * THREAD_N + 2u) * TILE_K + k]);
            let b3 = f32(tileB[(ty * THREAD_N + 3u) * TILE_K + k]);

            // Outer product: 4x4 accumulation
            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }

        workgroupBarrier();
    }

    // Write 4x4 output tile
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
