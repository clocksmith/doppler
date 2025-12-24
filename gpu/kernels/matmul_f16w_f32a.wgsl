// Matrix Multiplication Kernel - f16 weights, f32 activations
//
// A is f32 (activations), B is f16 (weights), C is f32.
// C[M,N] = A[M,K] * B[K,N]  (or B^T when transposeB=1)
// This reduces weight bandwidth and leverages f16 load/throughput
// while keeping f32 outputs for compatibility with f32-only ops.

enable f16;

const TILE_SIZE: u32 = 16u;

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    transposeB: u32,  // 0 = normal, 1 = B is stored transposed [N,K]
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f16>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

var<workgroup> tileA: array<f32, 256>;
var<workgroup> tileB: array<f16, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = gid.x;
    let col = gid.y;
    let local_row = lid.x;
    let local_col = lid.y;

    var sum: f32 = 0.0;

    let num_tiles = (uniforms.K + TILE_SIZE - 1u) / TILE_SIZE;
    let tile_idx = local_row * TILE_SIZE + local_col;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + local_col;
        let b_row = t * TILE_SIZE + local_row;

        if (row < uniforms.M && a_col < uniforms.K) {
            tileA[tile_idx] = A[row * uniforms.K + a_col];
        } else {
            tileA[tile_idx] = 0.0;
        }

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

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            let a_val = tileA[local_row * TILE_SIZE + k];
            let b_val = f32(tileB[k * TILE_SIZE + local_col]);
            sum = sum + a_val * b_val;
        }

        workgroupBarrier();
    }

    if (row < uniforms.M && col < uniforms.N) {
        C[row * uniforms.N + col] = sum * uniforms.alpha;
    }
}

