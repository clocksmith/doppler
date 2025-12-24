// Streaming Multi-Head Attention Kernel (no workgroup storage)
//
// Fallback variant for devices with extremely small shared memory or
// models with headDim beyond tiled support. Uses two-pass softmax and
// reads K/V directly from storage. Slower but compatible.

const MAX_HEAD_DIM: u32 = 256u;

struct AttentionUniforms {
    numHeads: u32,
    numKVHeads: u32,
    headDim: u32,
    seqLen: u32,
    queryLen: u32,
    scale: f32,
    isCausal: u32,
    startPos: u32,  // Absolute position offset for causal masking
}

@group(0) @binding(0) var<uniform> uniforms: AttentionUniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn getKVHeadIdx(queryHeadIdx: u32) -> u32 {
    let headsPerKV = uniforms.numHeads / uniforms.numKVHeads;
    return queryHeadIdx / headsPerKV;
}

fn isMasked(queryPos: u32, keyPos: u32) -> bool {
    if (uniforms.isCausal == 0u) { return false; }
    return keyPos > (queryPos + uniforms.startPos);
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let linear = wg_id.x;
    let numHeads = uniforms.numHeads;
    let headIdx = linear % numHeads;
    let queryPos = linear / numHeads;

    if (queryPos >= uniforms.queryLen) { return; }

    let kvHeadIdx = getKVHeadIdx(headIdx);
    let headDim = uniforms.headDim;
    let seqLen = uniforms.seqLen;
    let scale = uniforms.scale;

    var q_local: array<f32, 256>;
    let q_offset = queryPos * numHeads * headDim + headIdx * headDim;
    for (var d: u32 = 0u; d < headDim; d = d + 1u) {
        q_local[d] = Q[q_offset + d];
    }

    var maxScore: f32 = -3.402823e+38;
    for (var kPos: u32 = 0u; kPos < seqLen; kPos = kPos + 1u) {
        if (isMasked(queryPos, kPos)) { continue; }
        let k_offset = kPos * uniforms.numKVHeads * headDim + kvHeadIdx * headDim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < headDim; d = d + 1u) {
            dot = dot + q_local[d] * K[k_offset + d];
        }
        dot = dot * scale;
        maxScore = max(maxScore, dot);
    }

    var sumExp: f32 = 0.0;
    var acc: array<f32, 256>;
    for (var d: u32 = 0u; d < headDim; d = d + 1u) {
        acc[d] = 0.0;
    }

    for (var kPos: u32 = 0u; kPos < seqLen; kPos = kPos + 1u) {
        if (isMasked(queryPos, kPos)) { continue; }
        let k_offset = kPos * uniforms.numKVHeads * headDim + kvHeadIdx * headDim;
        let v_offset = kPos * uniforms.numKVHeads * headDim + kvHeadIdx * headDim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < headDim; d = d + 1u) {
            dot = dot + q_local[d] * K[k_offset + d];
        }
        dot = dot * scale;
        let w = exp(dot - maxScore);
        sumExp = sumExp + w;
        for (var d: u32 = 0u; d < headDim; d = d + 1u) {
            acc[d] = acc[d] + w * V[v_offset + d];
        }
    }

    if (sumExp <= 0.0) { return; }

    let out_offset = queryPos * numHeads * headDim + headIdx * headDim;
    for (var d: u32 = 0u; d < headDim; d = d + 1u) {
        output[out_offset + d] = acc[d] / sumExp;
    }
}

