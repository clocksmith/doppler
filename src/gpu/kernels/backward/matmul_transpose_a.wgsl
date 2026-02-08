// Matrix Multiplication Kernel - Transpose A
//
// C[M,N] = A^T[M,K] * B[K,N]
// Where A is stored as [K,M], so A^T is [M,K]
//
// This is used for matmul backward: dW = X^T * dY
// if X is [Batch, K] and dY is [Batch, N], then dW is [K, N]
// Here M = K_dim, K = Batch_dim, N = N_dim

const MAX_TILE_SIZE: u32 = 16u;
const TILE_AREA: u32 = MAX_TILE_SIZE * MAX_TILE_SIZE;

override TILE_SIZE: u32 = 16u;

struct Uniforms {
    M: u32,           // Rows of A^T and C
    N: u32,           // Cols of B and C
    K: u32,           // Cols of A^T, Rows of B
    alpha: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>; // Stored as [K, M]
@group(0) @binding(2) var<storage, read> B: array<f32>; // Stored as [K, N]
@group(0) @binding(3) var<storage, read_write> C: array<f32>; // Result [M, N]

var<workgroup> tileA: array<f32, TILE_AREA>;
var<workgroup> tileB: array<f32, TILE_AREA>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    let local_row = local_id.x;
    let local_col = local_id.y;

    var sum: f32 = 0.0;
    let num_tiles = (u.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let tile_k = t * TILE_SIZE;
        
        // Load tile A. A is [K, M]. We want A^T [M, K].
        // Element A^T[row, tile_k + local_col] is A[tile_k + local_col, row]
        let a_row_to_load = tile_k + local_col;
        let a_col_to_load = row;
        
        let tile_idx = local_row * TILE_SIZE + local_col;
        if (a_col_to_load < u.M && a_row_to_load < u.K) {
            tileA[tile_idx] = A[a_row_to_load * u.M + a_col_to_load];
        } else {
            tileA[tile_idx] = 0.0;
        }

        // Load tile B. B is [K, N].
        // Element B[tile_k + local_row, col]
        let b_row_to_load = tile_k + local_row;
        let b_col_to_load = col;
        if (b_row_to_load < u.K && b_col_to_load < u.N) {
            tileB[tile_idx] = B[b_row_to_load * u.N + b_col_to_load];
        } else {
            tileB[tile_idx] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum += tileA[local_row * TILE_SIZE + k] * tileB[k * TILE_SIZE + local_col];
        }

        workgroupBarrier();
    }

    if (row < u.M && col < u.N) {
        C[row * u.N + col] = sum * u.alpha;
    }
}
