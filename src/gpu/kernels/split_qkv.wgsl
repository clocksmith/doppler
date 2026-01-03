// split_qkv.wgsl

/**
 * Split fused QKV output into separate Q, K, V buffers.
 *
 * Input layout (row-major):
 *   [numTokens, qSize + kSize + vSize]
 *   Each row: [q_values, k_values, v_values]
 *
 * Output layout:
 *   Q: [numTokens, qSize]
 *   K: [numTokens, kSize]
 *   V: [numTokens, vSize]
 */

struct Params {
    numTokens: u32,
    qSize: u32,
    kSize: u32,
    vSize: u32,
}

override WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> Q: array<f32>;
@group(0) @binding(3) var<storage, read_write> K: array<f32>;
@group(0) @binding(4) var<storage, read_write> V: array<f32>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let numTokens = params.numTokens;
    let qSize = params.qSize;
    let kSize = params.kSize;
    let vSize = params.vSize;
    let qkvSize = qSize + kSize + vSize;

    // Each thread handles one element from Q, K, or V
    let totalQ = numTokens * qSize;
    let totalK = numTokens * kSize;
    let totalV = numTokens * vSize;
    let totalElements = totalQ + totalK + totalV;

    if (idx >= totalElements) {
        return;
    }

    if (idx < totalQ) {
        // Q output
        let token = idx / qSize;
        let elem = idx % qSize;
        let srcIdx = token * qkvSize + elem;
        Q[idx] = input[srcIdx];
    } else if (idx < totalQ + totalK) {
        // K output
        let kIdx = idx - totalQ;
        let token = kIdx / kSize;
        let elem = kIdx % kSize;
        let srcIdx = token * qkvSize + qSize + elem;
        K[kIdx] = input[srcIdx];
    } else {
        // V output
        let vIdx = idx - totalQ - totalK;
        let token = vIdx / vSize;
        let elem = vIdx % vSize;
        let srcIdx = token * qkvSize + qSize + kSize + elem;
        V[vIdx] = input[srcIdx];
    }
}
