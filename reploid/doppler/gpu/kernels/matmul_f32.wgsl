// Matrix Multiplication Kernel - FP32 Fallback
//
// Tiled matrix multiplication using shared memory (workgroup storage)
// C[M,N] = A[M,K] * B[K,N]  (or B^T when transposeB=1)
//
// This is the fallback kernel when shader-f16 is unavailable.
// Uses 16x16 tiles for good occupancy across devices.

// Tile dimensions - optimized for 256 threads per workgroup
const TILE_SIZE: u32 = 16u;

// Uniforms for matrix dimensions
struct Uniforms {
    M: u32,  // Rows of A and C
    N: u32,  // Cols of B and C (or rows of B when transposed)
    K: u32,  // Cols of A, Rows of B (or cols of B when transposed)
    alpha: f32,  // Scaling factor (typically 1.0)
    transposeB: u32,  // 0 = normal, 1 = B is stored transposed [N,K] -> treat as [K,N]
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

// Shared memory tiles for A and B
var<workgroup> tileA: array<f32, 256>;  // TILE_SIZE * TILE_SIZE
var<workgroup> tileB: array<f32, 256>;  // TILE_SIZE * TILE_SIZE

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    let local_row = local_id.x;
    let local_col = local_id.y;

    // Accumulator for dot product
    var sum: f32 = 0.0;

    // Number of tiles needed to cover K dimension
    let num_tiles = (uniforms.K + TILE_SIZE - 1u) / TILE_SIZE;

    // Iterate over tiles
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Global indices for loading tiles
        let a_col = t * TILE_SIZE + local_col;
        let b_row = t * TILE_SIZE + local_row;

        // Load tile from A into shared memory (with bounds check)
        let tile_idx = local_row * TILE_SIZE + local_col;
        if (row < uniforms.M && a_col < uniforms.K) {
            tileA[tile_idx] = A[row * uniforms.K + a_col];
        } else {
            tileA[tile_idx] = 0.0;
        }

        // Load tile from B into shared memory (with bounds check)
        // When transposeB=1, B is stored as [N,K] so we access B[col,b_row] instead of B[b_row,col]
        if (b_row < uniforms.K && col < uniforms.N) {
            if (uniforms.transposeB == 0u) {
                tileB[tile_idx] = B[b_row * uniforms.N + col];
            } else {
                // B is [N, K], access element [col, b_row]
                tileB[tile_idx] = B[col * uniforms.K + b_row];
            }
        } else {
            tileB[tile_idx] = 0.0;
        }

        // Synchronize to ensure tile is fully loaded
        workgroupBarrier();

        // Compute partial dot product for this tile
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tileA[local_row * TILE_SIZE + k] * tileB[k * TILE_SIZE + local_col];
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Write result (with bounds check)
    if (row < uniforms.M && col < uniforms.N) {
        C[row * uniforms.N + col] = sum * uniforms.alpha;
    }
}
