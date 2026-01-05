// Matrix Multiplication Kernel - FP16 (Half Precision)
//
// Tiled matrix multiplication using FP16 for improved throughput.
// Requires 'shader-f16' feature enabled.
// C[M,N] = A[M,K] * B[K,N]  (or B^T when transpose_b=1)
//
// Uses FP16 for storage and computation, with optional FP32 accumulation
// for better numerical stability.

enable f16;

// Tile dimensions - can use larger tiles with f16 due to smaller footprint
const TILE_SIZE: u32 = 16u; // Must be const because it's used in workgroup array sizes.
const TILE_AREA: u32 = TILE_SIZE * TILE_SIZE;

// Uniforms for matrix dimensions
struct Uniforms {
    M: u32,           // Rows of A and C
    N: u32,           // Cols of B and C (or rows of B when transposed)
    K: u32,           // Cols of A, Rows of B (or cols of B when transposed)
    alpha: f32,       // Scaling factor
    transpose_b: u32, // 0 = normal, 1 = B is stored transposed [N,K] -> treat as [K,N]
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f16>;
@group(0) @binding(2) var<storage, read> B: array<f16>;
@group(0) @binding(3) var<storage, read_write> C: array<f16>;

// Shared memory tiles - f16 allows 2x data in same space
var<workgroup> tileA: array<f16, TILE_AREA>;
var<workgroup> tileB: array<f16, TILE_AREA>;
var<workgroup> tileB_vec4: array<vec4<f16>, TILE_AREA>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
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

    let num_tiles = (u.K + TILE_SIZE - 1u) / TILE_SIZE;
    let tile_idx = local_row * TILE_SIZE + local_col;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + local_col;
        let b_row = t * TILE_SIZE + local_row;

        // Load tile from A
        if (row < u.M && a_col < u.K) {
            tileA[tile_idx] = A[row * u.K + a_col];
        } else {
            tileA[tile_idx] = f16(0.0);
        }

        // Load tile from B (handle transpose)
        if (b_row < u.K && col < u.N) {
            if (u.transpose_b == 0u) {
                tileB[tile_idx] = B[b_row * u.N + col];
            } else {
                // B is [N, K], access element [col, b_row]
                tileB[tile_idx] = B[col * u.K + b_row];
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
    if (row < u.M && col < u.N) {
        C[row * u.N + col] = f16(sum * u.alpha);
    }
}

// Alternative entry point for vec4 column groups (best when N is a multiple of 4)
@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main_vec4(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.x;
    let col_base = global_id.y * 4u;  // Each thread handles 4 columns
    let local_row = local_id.x;
    let local_col = local_id.y;

    var sum: vec4<f32> = vec4<f32>(0.0);

    let num_tiles = (u.K + TILE_SIZE - 1u) / TILE_SIZE;
    let tile_idx = local_row * TILE_SIZE + local_col;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + local_col;
        let b_row = t * TILE_SIZE + local_row;

        // Load A tile (single element per thread)
        if (row < u.M && a_col < u.K) {
            tileA[tile_idx] = A[row * u.K + a_col];
        } else {
            tileA[tile_idx] = f16(0.0);
        }

        // Load B tile (handle transpose)
        var b_vec: vec4<f16> = vec4<f16>(f16(0.0));
        if (b_row < u.K) {
            if (u.transpose_b == 0u) {
                if (col_base + 0u < u.N) { b_vec.x = B[b_row * u.N + col_base]; }
                if (col_base + 1u < u.N) { b_vec.y = B[b_row * u.N + col_base + 1u]; }
                if (col_base + 2u < u.N) { b_vec.z = B[b_row * u.N + col_base + 2u]; }
                if (col_base + 3u < u.N) { b_vec.w = B[b_row * u.N + col_base + 3u]; }
            } else {
                if (col_base + 0u < u.N) { b_vec.x = B[(col_base + 0u) * u.K + b_row]; }
                if (col_base + 1u < u.N) { b_vec.y = B[(col_base + 1u) * u.K + b_row]; }
                if (col_base + 2u < u.N) { b_vec.z = B[(col_base + 2u) * u.K + b_row]; }
                if (col_base + 3u < u.N) { b_vec.w = B[(col_base + 3u) * u.K + b_row]; }
            }
        }
        tileB_vec4[tile_idx] = b_vec;

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            let a_val = f32(tileA[local_row * TILE_SIZE + k]);
            let b_val = tileB_vec4[k * TILE_SIZE + local_col];
            sum = sum + a_val * vec4<f32>(b_val);
        }

        workgroupBarrier();
    }

    // Write results
    if (row < u.M && col_base < u.N) {
        let base = row * u.N + col_base;
        let scaled = sum * u.alpha;
        if (col_base + 0u < u.N) { C[base] = f16(scaled.x); }
        if (col_base + 1u < u.N) { C[base + 1u] = f16(scaled.y); }
        if (col_base + 2u < u.N) { C[base + 2u] = f16(scaled.z); }
        if (col_base + 3u < u.N) { C[base + 3u] = f16(scaled.w); }
    }
}
