// Optimized GEMV Kernel - Parallel Reduction over K
// For M=1 decode: C[N] = A[K] * B^T[K,N]
//
// Key optimizations:
// 1. Each workgroup computes ONE output element C[col]
// 2. Threads cooperatively reduce over K dimension (parallel reduction)
// 3. Coalesced memory access - adjacent threads read adjacent K elements
// 4. Final reduction in shared memory
//
// A is f32 (activations), B is f16 (weights transposed [N,K]), C is f32.

enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    M: u32,           // Always 1 for GEMV
    N: u32,           // Output dimension (# of output columns)
    K: u32,           // Inner dimension (dot product length)
    alpha: f32,       // Scaling factor
    transpose_b: u32, // Expected to be 1 (B stored as [N,K])
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f16>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

// Shared memory for parallel reduction
var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let col = wg_id.x;  // Output column this workgroup computes
    let local_id = lid.x;

    if (col >= u.N) {
        return;
    }

    // Each thread computes partial sum for k = local_id, local_id+WORKGROUP_SIZE, ...
    var partial_sum: f32 = 0.0;

    // B is stored transposed [N, K], so B[col, k] = B[col * K + k]
    let b_row_offset = col * u.K;

    // Stride through K with workgroup-sized steps
    var k: u32 = local_id;
    for (; k < u.K; k = k + WORKGROUP_SIZE) {
        let a_val = A[k];
        let b_val = f32(B[b_row_offset + k]);
        partial_sum = partial_sum + a_val * b_val;
    }

    // Store partial sum to shared memory
    shared_sum[local_id] = partial_sum;
    workgroupBarrier();

    // Parallel reduction in shared memory
    // Tree reduction: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (local_id < stride) {
            shared_sum[local_id] = shared_sum[local_id] + shared_sum[local_id + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes the final result
    if (local_id == 0u) {
        C[col] = shared_sum[0] * u.alpha;
    }
}
