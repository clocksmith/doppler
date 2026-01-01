// Naive Matrix Multiplication Kernel - f16 weights, f32 activations
// For M=1 decode case - simple dot product per output element
//
// A is f32 (activations), B is f16 (weights), C is f32.
// C[1,N] = A[1,K] * B[K,N] (or B^T when transpose_b=1)
// Each thread computes one output element via dot product.

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    transpose_b: u32,  // 0 = normal, 1 = B is stored transposed [N,K]
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f16>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;

    if (col >= u.N) {
        return;
    }

    // For M=1, compute dot product: C[0, col] = sum_k A[0, k] * B[k, col]
    var sum: f32 = 0.0;

    for (var k: u32 = 0u; k < u.K; k = k + 1u) {
        let a_val = A[k];  // A[0, k]

        var b_val: f16;
        if (u.transpose_b == 0u) {
            b_val = B[k * u.N + col];  // B[k, col]
        } else {
            // B is [N, K], access element [col, k]
            b_val = B[col * u.K + k];
        }

        sum = sum + a_val * f32(b_val);
    }

    C[col] = sum * u.alpha;
}
