// Matrix Multiplication Backward Kernel - dX = dY @ W^T
//
// Fused tiled matmul for computing input gradients during backpropagation.
//
// When transpose_b = 1 (standard case: forward was Y = X @ W, W stored [K,N]):
//   dX[M,K] = dY[M,N] @ W^T[N,K]   (reads W as [K,N], transposes on the fly)
//
// When transpose_b = 0 (forward was Y = X @ W^T, W stored [N,K]):
//   dX[M,K] = dY[M,N] @ W[N,K]     (reads W directly as [N,K])
//
// Uses 16x16 shared memory tiles, matching matmul_transpose_a.wgsl style.

const MAX_TILE_SIZE: u32 = 16u;
const TILE_AREA: u32 = MAX_TILE_SIZE * MAX_TILE_SIZE;

override TILE_SIZE: u32 = 16u;

struct Uniforms {
    M: u32,            // Rows of dY and dX (batch dim)
    N: u32,            // Cols of dY (output dim of forward)
    K: u32,            // Cols of dX (input dim of forward)
    alpha: f32,
    transpose_b: u32,  // 0 = W stored [N,K], 1 = W stored [K,N]
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> dY: array<f32>;  // [M, N]
@group(0) @binding(2) var<storage, read> W: array<f32>;   // [K, N] or [N, K]
@group(0) @binding(3) var<storage, read_write> dX: array<f32>; // [M, K]

var<workgroup> tileDY: array<f32, TILE_AREA>;
var<workgroup> tileW: array<f32, TILE_AREA>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.x;  // M dim
    let col = global_id.y;  // K dim
    let local_row = local_id.x;
    let local_col = local_id.y;

    var sum: f32 = 0.0;
    let num_tiles = (u.N + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let tile_n = t * TILE_SIZE;
        let tile_idx = local_row * TILE_SIZE + local_col;

        // Load tile from dY [M, N].
        // Element dY[row, tile_n + local_col]
        let dy_col = tile_n + local_col;
        if (row < u.M && dy_col < u.N) {
            tileDY[tile_idx] = dY[row * u.N + dy_col];
        } else {
            tileDY[tile_idx] = 0.0;
        }

        // Load tile from W^T [N, K].
        // We need W^T[tile_n + local_row, col].
        let wt_row = tile_n + local_row;
        if (wt_row < u.N && col < u.K) {
            if (u.transpose_b == 0u) {
                // W stored as [N, K]: W^T[wt_row, col] = W[wt_row, col]
                tileW[tile_idx] = W[wt_row * u.K + col];
            } else {
                // W stored as [K, N]: W^T[wt_row, col] = W[col, wt_row]
                tileW[tile_idx] = W[col * u.N + wt_row];
            }
        } else {
            tileW[tile_idx] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum += tileDY[local_row * TILE_SIZE + k] * tileW[k * TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    if (row < u.M && col < u.K) {
        dX[row * u.K + col] = sum * u.alpha;
    }
}
