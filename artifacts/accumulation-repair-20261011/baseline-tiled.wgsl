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

            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }

        workgroupBarrier();
    }

    // Write 4x4 output tile (f32)
    let alpha = u.alpha;
    if (row_base + 0u < u.M && col_base + 0u < u.N) { C[(row_base + 0u) * u.N + col_base + 0u] = acc00 * alpha; }
    if (row_base + 0u < u.M && col_base + 1u < u.N) { C[(row_base + 0u) * u.N + col_base + 1u] = acc01 * alpha; }
    if (row_base + 0u < u.M && col_base + 2u < u.N) { C[(row_base + 0u) * u.N + col_base + 2u] = acc02 * alpha; }
    if (row_base + 0u < u.M && col_base + 3u < u.N) { C[(row_base + 0u) * u.N + col_base + 3u] = acc03 * alpha; }

    if (row_base + 1u < u.M && col_base + 0u < u.N) { C[(row_base + 1u) * u.N + col_base + 0u] = acc10 * alpha; }
    if (row_base + 1u < u.M && col_base + 1u < u.N) { C[(row_base + 1u) * u.N + col_base + 1u] = acc11 * alpha; }
    if (row_base + 1u < u.M && col_base + 2u < u.N) { C[(row_base + 1u) * u.N + col_base + 2u] = acc12 * alpha; }
    if (row_base + 1u < u.M && col_base + 3u < u.N) { C[(row_base + 1u) * u.N + col_base + 3u] = acc13 * alpha; }

    if (row_base + 2u < u.M && col_base + 0u < u.N) { C[(row_base + 2u) * u.N + col_base + 0u] = acc20 * alpha; }
    if (row_base + 2u < u.M && col_base + 1u < u.N) { C[(row_base + 2u) * u.N + col_base + 1u] = acc21 * alpha; }
    if (row_base + 2u < u.M && col_base + 2u < u.N) { C[(row_base + 2u) * u.N + col_base + 2u] = acc22 * alpha; }
    if (row_base + 2u < u.M && col_base + 3u < u.N) { C[(row_base + 2u) * u.N + col_base + 3u] = acc23 * alpha; }

    if (row_base + 3u < u.M && col_base + 0u < u.N) { C[(row_base + 3u) * u.N + col_base + 0u] = acc30 * alpha; }
    if (row_base + 3u < u.M && col_base + 1u < u.N) { C[(row_base + 3u) * u.N + col_base + 1u] = acc31 * alpha; }
    if (row_base + 3u < u.M && col_base + 2u < u.N) { C[(row_base + 3u) * u.N + col_base + 2u] = acc32 * alpha; }
    if (row_base + 3u < u.M && col_base + 3u < u.N) { C[(row_base + 3u) * u.N + col_base + 3u] = acc33 * alpha; }
}
