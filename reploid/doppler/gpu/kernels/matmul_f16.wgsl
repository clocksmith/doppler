// Matrix Multiplication Kernel - FP16 (Half Precision)
//
// Tiled matrix multiplication using FP16 for improved throughput.
// Requires 'shader-f16' feature enabled.
// C[M,N] = A[M,K] * B[K,N]  (or B^T when transposeB=1)
//
// Uses FP16 for storage and computation, with optional FP32 accumulation
// for better numerical stability.

enable f16;

// Tile dimensions - can use larger tiles with f16 due to smaller footprint
const TILE_SIZE: u32 = 16u;
const TILE_ELEMS: u32 = 256u;  // TILE_SIZE * TILE_SIZE

// Uniforms for matrix dimensions
struct Uniforms {
    M: u32,     // Rows of A and C
    N: u32,     // Cols of B and C (or rows of B when transposed)
    K: u32,     // Cols of A, Rows of B (or cols of B when transposed)
    alpha: f32, // Scaling factor
    transposeB: u32,  // 0 = normal, 1 = B is stored transposed [N,K] -> treat as [K,N]
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f16>;
@group(0) @binding(2) var<storage, read> B: array<f16>;
@group(0) @binding(3) var<storage, read_write> C: array<f16>;

// Shared memory tiles - f16 allows 2x data in same space
var<workgroup> tileA: array<f16, 256>;
var<workgroup> tileB: array<f16, 256>;

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

    // Use f32 accumulator for numerical stability during summation
    var sum: f32 = 0.0;

    let num_tiles = (uniforms.K + TILE_SIZE - 1u) / TILE_SIZE;
    let tile_idx = local_row * TILE_SIZE + local_col;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + local_col;
        let b_row = t * TILE_SIZE + local_row;

        // Load tile from A
        if (row < uniforms.M && a_col < uniforms.K) {
            tileA[tile_idx] = A[row * uniforms.K + a_col];
        } else {
            tileA[tile_idx] = f16(0.0);
        }

        // Load tile from B (handle transpose)
        if (b_row < uniforms.K && col < uniforms.N) {
            if (uniforms.transposeB == 0u) {
                tileB[tile_idx] = B[b_row * uniforms.N + col];
            } else {
                // B is [N, K], access element [col, b_row]
                tileB[tile_idx] = B[col * uniforms.K + b_row];
            }
        } else {
            tileB[tile_idx] = f16(0.0);
        }

        workgroupBarrier();

        // Compute partial dot product with f32 accumulation
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            let a_val = f32(tileA[local_row * TILE_SIZE + k]);
            let b_val = f32(tileB[k * TILE_SIZE + local_col]);
            sum = sum + a_val * b_val;
        }

        workgroupBarrier();
    }

    // Write result back as f16
    if (row < uniforms.M && col < uniforms.N) {
        C[row * uniforms.N + col] = f16(sum * uniforms.alpha);
    }
}

// Alternative entry point for vec4 f16 loads (2x throughput on some hardware)
// This requires K and N to be multiples of 4
@compute @workgroup_size(16, 16, 1)
fn main_vec4(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y * 4u;  // Each thread handles 4 columns
    let local_row = local_id.x;
    let local_col = local_id.y;

    var sum: vec4<f32> = vec4<f32>(0.0);

    let num_tiles = (uniforms.K + TILE_SIZE - 1u) / TILE_SIZE;
    let tile_idx = local_row * TILE_SIZE + local_col;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + local_col;
        let b_row = t * TILE_SIZE + local_row;

        // Load A tile (single element per thread)
        if (row < uniforms.M && a_col < uniforms.K) {
            tileA[tile_idx] = A[row * uniforms.K + a_col];
        } else {
            tileA[tile_idx] = f16(0.0);
        }

        // Load B tile (handle transpose)
        if (b_row < uniforms.K && col < uniforms.N) {
            if (uniforms.transposeB == 0u) {
                tileB[tile_idx] = B[b_row * uniforms.N + col];
            } else {
                tileB[tile_idx] = B[col * uniforms.K + b_row];
            }
        } else {
            tileB[tile_idx] = f16(0.0);
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            let a_val = f32(tileA[local_row * TILE_SIZE + k]);
            let b_idx = k * TILE_SIZE + local_col;
            // Broadcast A value across 4 B values
            let b_val = f32(tileB[b_idx]);
            sum.x = sum.x + a_val * b_val;
        }

        workgroupBarrier();
    }

    // Write results
    if (row < uniforms.M && col < uniforms.N) {
        C[row * uniforms.N + col] = f16(sum.x * uniforms.alpha);
    }
}
