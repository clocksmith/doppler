// Matrix Multiplication Kernel - FP32 Fallback
//
// Tiled matrix multiplication using shared memory (workgroup storage)
// C[M,N] = A[M,K] * B[K,N]  (or B^T when transpose_b=1)
//
// This is the fallback kernel when shader-f16 is unavailable.
// Uses 16x16 tiles for good occupancy across devices.

// Tile dimensions - optimized for 256 threads per workgroup
const MAX_TILE_SIZE: u32 = 16u;
const TILE_STRIDE: u32 = MAX_TILE_SIZE + 1u;
const TILE_AREA: u32 = MAX_TILE_SIZE * TILE_STRIDE;

override TILE_SIZE: u32 = 16u;  // Must be <= MAX_TILE_SIZE

// Uniforms for matrix dimensions
struct Uniforms {
    M: u32,           // Rows of A and C
    N: u32,           // Cols of B and C (or rows of B when transposed)
    K: u32,           // Cols of A, Rows of B (or cols of B when transposed)
    alpha: f32,       // Scaling factor (typically 1.0)
    transpose_b: u32, // 0 = normal, 1 = B is stored transposed [N,K] -> treat as [K,N]
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

// Shared memory tiles for A and B
var<workgroup> tileA: array<f32, TILE_AREA>;
var<workgroup> tileB: array<f32, TILE_AREA>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    if (TILE_SIZE > MAX_TILE_SIZE) {
        return;
    }

    let local_col = local_id.x;
    let local_row = local_id.y;
    let row = workgroup_id.x * TILE_SIZE + local_row;
    let col = workgroup_id.y * TILE_SIZE + local_col;

    // Accumulator for dot product
    var sum: f32 = 0.0;

    // Number of tiles needed to cover K dimension
    let num_tiles = (u.K + TILE_SIZE - 1u) / TILE_SIZE;

    // Iterate over tiles
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Global indices for loading tiles (local_col = local_id.x is contiguous across lanes)
        let a_col = t * TILE_SIZE + local_col;

        // Load tile from A into shared memory (coalesced across local_col)
        let tile_idx = local_row * TILE_STRIDE + local_col;
        if (row < u.M && a_col < u.K) {
            tileA[tile_idx] = A[row * u.K + a_col];
        } else {
            tileA[tile_idx] = 0.0;
        }

        // Load tile from B into shared memory (coalesced across local_col in both layouts)
        if (u.transpose_b == 0u) {
            let b_row = t * TILE_SIZE + local_row;
            if (b_row < u.K && col < u.N) {
                tileB[tile_idx] = B[b_row * u.N + col];
            } else {
                tileB[tile_idx] = 0.0;
            }
        } else {
            // B is [N, K]; coalesce across K using local_col and transpose into padded tileB.
            let b_col = workgroup_id.y * TILE_SIZE + local_row;
            let b_k = a_col;
            if (b_k < u.K && b_col < u.N) {
                tileB[local_col * TILE_STRIDE + local_row] = B[b_col * u.K + b_k];
            } else {
                tileB[local_col * TILE_STRIDE + local_row] = 0.0;
            }
        }

        // Synchronize to ensure tile is fully loaded
        workgroupBarrier();

        // Compute partial dot product for this tile
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tileA[local_row * TILE_STRIDE + k] * tileB[k * TILE_STRIDE + local_col];
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Write result (with bounds check)
    if (row < u.M && col < u.N) {
        C[row * u.N + col] = sum * u.alpha;
    }
}
