// Naive Matrix Multiplication Kernel - f16 weights, f32 activations
// For M=1 decode case - simple dot product per output element
//
// A is f32 (activations), B is f16 (weights), C is f32.
// C[1,N] = A[1,K] * B[K,N] (or B^T when transposeB=1)
// Each thread computes one output element via dot product.

enable f16;

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

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;

    if (col >= uniforms.N) {
        return;
    }

    // For M=1, compute dot product: C[0, col] = sum_k A[0, k] * B[k, col]
    var sum: f32 = 0.0;

    for (var k: u32 = 0u; k < uniforms.K; k = k + 1u) {
        let a_val = A[k];  // A[0, k]

        var b_val: f16;
        if (uniforms.transposeB == 0u) {
            b_val = B[k * uniforms.N + col];  // B[k, col]
        } else {
            // B is [N, K], access element [col, k]
            b_val = B[col * uniforms.K + k];
        }

        sum = sum + a_val * f32(b_val);
    }

    C[col] = sum * uniforms.alpha;
}
