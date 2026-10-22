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

// Shared memory, k-major: tileA[k * WG_M + tx] holds rows tx*4..tx*4+3 of column k,
// tileB[k * WG_N + ty] holds weight rows ty*4..ty*4+3 of column k. Each inner-loop
// step reads one vec4 per operand: A is broadcast across ty, and B is contiguous
// across ty, so neither read bank-conflicts. 64*16 f32 per tile → 8192 bytes total.
// Each thread stages exactly one whole vec4 per tile. Threads must never write
// separate components of one shared vec4: a dynamically indexed component store
// can lower to a read-modify-write of the whole vector (Metal does), which races.
var<workgroup> tileA: array<vec4<f32>, 256>;  // TILE_K * WG_M
var<workgroup> tileB: array<vec4<f32>, 256>;  // TILE_K * WG_N

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

// Guarded global reads; out-of-range elements stage as zero.
fn load_a(row: u32, col: u32) -> f32 {
    if (row < u.M && col < u.K) { return A[row * u.K + col]; }
    return 0.0;
}

fn load_b(n: u32, col: u32) -> f32 {
    if (n < u.N && col < u.K) { return f32(B[n * u.K + col]); }
    return 0.0;
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

    let num_tiles = (u.K + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let k_offset = t * TILE_K;

        // Load A tile (f32): 64 rows x 16 cols. Thread tid stages column k = tid % 16 of
        // rows group*4..group*4+3, group = tid / 16, as one vec4. Sixteen consecutive
        // threads read sixteen consecutive floats of each row.
        {
            let group = tid / TILE_K;
            let k = tid % TILE_K;
            let row = wg_id.x * TILE_M + group * THREAD_M;
            let col = k_offset + k;
            tileA[k * WG_M + group] = vec4<f32>(
                load_a(row, col), load_a(row + 1u, col), load_a(row + 2u, col), load_a(row + 3u, col)
            );
        }

        // Load B tile (f16 → f32): B is [N, K] (row-major/transposed), staged the same way.
        {
            let group = tid / TILE_K;
            let k = tid % TILE_K;
            let n = wg_id.y * TILE_N + group * THREAD_N;
            let col = k_offset + k;
            tileB[k * WG_N + group] = vec4<f32>(
                load_b(n, col), load_b(n + 1u, col), load_b(n + 2u, col), load_b(n + 3u, col)
            );
        }

        workgroupBarrier();

        // Bound each local accumulation to one K tile. Keep only the tile
        // subtotal and the outer sum for each output, all in F32.
        var tile00: f32 = 0.0; var tile01: f32 = 0.0; var tile02: f32 = 0.0; var tile03: f32 = 0.0;
        var tile10: f32 = 0.0; var tile11: f32 = 0.0; var tile12: f32 = 0.0; var tile13: f32 = 0.0;
        var tile20: f32 = 0.0; var tile21: f32 = 0.0; var tile22: f32 = 0.0; var tile23: f32 = 0.0;
        var tile30: f32 = 0.0; var tile31: f32 = 0.0; var tile32: f32 = 0.0; var tile33: f32 = 0.0;

        // Compute: each thread accumulates its 4x4 output tile
        for (var k: u32 = 0u; k < TILE_K; k = k + 1u) {
            let a = tileA[k * WG_M + tx];
            let b = tileB[k * WG_N + ty];
            let a0 = a.x; let a1 = a.y; let a2 = a.z; let a3 = a.w;
            let b0 = b.x; let b1 = b.y; let b2 = b.z; let b3 = b.w;

            tile00 = fma(a0, b0, tile00); tile01 = fma(a0, b1, tile01); tile02 = fma(a0, b2, tile02); tile03 = fma(a0, b3, tile03);
            tile10 = fma(a1, b0, tile10); tile11 = fma(a1, b1, tile11); tile12 = fma(a1, b2, tile12); tile13 = fma(a1, b3, tile13);
            tile20 = fma(a2, b0, tile20); tile21 = fma(a2, b1, tile21); tile22 = fma(a2, b2, tile22); tile23 = fma(a2, b3, tile23);
            tile30 = fma(a3, b0, tile30); tile31 = fma(a3, b1, tile31); tile32 = fma(a3, b2, tile32); tile33 = fma(a3, b3, tile33);
        }

        acc00 += tile00; acc01 += tile01; acc02 += tile02; acc03 += tile03;
        acc10 += tile10; acc11 += tile11; acc12 += tile12; acc13 += tile13;
        acc20 += tile20; acc21 += tile21; acc22 += tile22; acc23 += tile23;
        acc30 += tile30; acc31 += tile31; acc32 += tile32; acc33 += tile33;

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
